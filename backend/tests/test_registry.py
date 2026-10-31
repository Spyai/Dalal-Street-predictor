"""Model registry: save / champion / promote with the real database, promotion rules, MLflow.

The promotion rule is exercised with hand-built fold metrics: a champion trained through
30 Oct 2026 and challengers whose folds are better or worse inside and outside the
champion's last 182 days, so each rule (beat on the common window, data refresh,
whole-sample fallback) is hit on purpose.
"""

from __future__ import annotations

import re
import sys
import types
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml import registry
from app.ml.core import TrainResult, WalkForwardFold, WalkForwardMetrics
from app.models import ModelVersion

TARGET = "NIFTY 50"
TRAINED_THROUGH = date(2026, 10, 30)
WINDOW_START = TRAINED_THROUGH - timedelta(days=registry.WINDOW_DAYS)  # 2026-05-01
FEATURES = ["ret_1d", "es_fut_ret_1d"]
RECIPE: dict[str, object] = {"seed": 42, "num_leaves": 5, "lambda_l2": 5.0}


def _metrics(
    brier: float, accuracy: float, folds: list[WalkForwardFold] | None = None
) -> WalkForwardMetrics:
    return WalkForwardMetrics(
        n_oos=200,
        accuracy=accuracy,
        brier=brier,
        log_loss=1.05,
        class_frequencies={"UP": 0.4, "FLAT": 0.25, "DOWN": 0.35},
        baselines={"always_up": {"n": 200, "accuracy": 0.4, "brier": 1.2}},
        by_confidence=[{"confidence": "high", "n": 50, "accuracy": 0.5}],
        calibration=[
            {"bin_lo": 0.3333, "bin_hi": 0.4667, "n": 50, "predicted_mean": 0.4, "observed": 0.42}
        ],
        folds=folds or [],
    )


def _result(
    tag: str,
    brier: float = 0.64,
    accuracy: float = 0.5,
    folds: list[WalkForwardFold] | None = None,
    *,
    params: dict[str, object] | None = None,
    feature_names: list[str] | None = None,
    train_rows: int = 1200,
    trained_through: date = TRAINED_THROUGH,
) -> TrainResult:
    return TrainResult(
        artifact=f'{{"classifier": "{tag}", "q10": "", "q90": ""}}',
        feature_names=list(feature_names or FEATURES),
        params=dict(
            params if params is not None else {**RECIPE, "n_estimators": 100, "shrink": 0.3}
        ),
        metrics=_metrics(brier, accuracy, folds),
        trained_through=trained_through,
        train_rows=train_rows,
    )


def _folds(
    briers: list[float], *, end: date = TRAINED_THROUGH, n_test: int = 21
) -> list[WalkForwardFold]:
    """Consecutive 21-session folds ending at `end`, newest last, each 30 calendar days long."""
    out: list[WalkForwardFold] = []
    test_end = end
    for b in reversed(briers):
        test_start = test_end - timedelta(days=29)
        out.append(
            WalkForwardFold(
                train_end=test_start - timedelta(days=1),
                test_start=test_start,
                test_end=test_end,
                n_test=n_test,
                accuracy=0.4,
                brier=b,
                log_loss=1.05,
            )
        )
        test_end = test_start - timedelta(days=1)
    return list(reversed(out))


def _champion_row(result: TrainResult) -> ModelVersion:
    """An unsaved ModelVersion carrying what `save_model` would store."""
    return ModelVersion(
        name="market_call",
        target=TARGET,
        version="lgbm-test",
        kind="lightgbm",
        params=dict(result.params),
        feature_names=list(result.feature_names),
        metrics=result.metrics.as_dict(),
        artifact=result.artifact,
        trained_through=result.trained_through,
        train_rows=result.train_rows,
        is_champion=True,
    )


def _champions(db: Session, target: str = TARGET) -> list[str]:
    db.expire_all()
    rows = db.execute(
        text("SELECT version FROM model_version WHERE is_champion AND target = :t ORDER BY id"),
        {"t": target},
    ).scalars()
    return list(rows)


# ------------------------------------------------------------------ save / champion / promote


def test_save_model_and_champion(db: Session) -> None:
    assert registry.champion(db, TARGET) is None
    mv = registry.save_model(db, TARGET, _result("a"), promote=True, note="first model")
    db.commit()
    assert re.fullmatch(r"lgbm-\d{8}T\d{4}Z-[0-9a-f]{6}", mv.version), mv.version
    assert mv.kind == "lightgbm" and mv.name == "market_call" and mv.target == TARGET
    assert mv.is_champion and mv.promoted_at is not None and mv.promotion_note == "first model"
    assert mv.feature_names == ["ret_1d", "es_fut_ret_1d"]
    assert mv.metrics["brier"] == 0.64 and mv.metrics["class_frequencies"]["UP"] == 0.4
    assert mv.trained_through == date(2026, 10, 30) and mv.train_rows == 1200
    assert mv.artifact == _result("a").artifact
    champ = registry.champion(db, TARGET)
    assert champ is not None and champ.id == mv.id
    assert registry.champion(db, "NIFTY BANK") is None  # champions are per target


def test_second_promote_demotes_first(db: Session) -> None:
    first = registry.save_model(db, TARGET, _result("a"), promote=True)
    second = registry.save_model(db, TARGET, _result("b"), promote=False)
    db.commit()
    assert not second.is_champion and second.promoted_at is None
    assert _champions(db) == [first.version]

    registry.promote(db, second, "beat the champion on Brier")
    db.commit()
    db.expire_all()
    assert _champions(db) == [second.version]
    assert registry.champion_count(db, TARGET) == 1
    champ = registry.champion(db, TARGET)
    assert champ is not None and champ.id == second.id
    assert champ.promotion_note == "beat the champion on Brier"
    assert db.get(ModelVersion, first.id).is_champion is False  # type: ignore[union-attr]

    # Promoting the champion again is a no-op for uniqueness.
    registry.promote(db, second, "again")
    db.commit()
    assert _champions(db) == [second.version]


def test_unique_champion_index_holds(db: Session) -> None:
    first = registry.save_model(db, TARGET, _result("a"), promote=True)
    second = registry.save_model(db, TARGET, _result("b"), promote=False)
    db.commit()
    with pytest.raises(IntegrityError), db.begin_nested():
        db.execute(
            text("UPDATE model_version SET is_champion = true WHERE id = :id"), {"id": second.id}
        )
    assert _champions(db) == [first.version]
    # A champion for another target does not collide.
    registry.save_model(db, "NIFTY BANK", _result("c"), promote=True)
    db.commit()
    assert _champions(db, "NIFTY BANK") and _champions(db) == [first.version]


def test_same_artifact_twice_in_a_minute_gets_distinct_versions(db: Session) -> None:
    a = registry.save_model(db, TARGET, _result("same"), promote=True)
    b = registry.save_model(db, TARGET, _result("same"), promote=False)
    db.commit()
    assert a.version != b.version and b.version.startswith(a.version)
    assert _champions(db) == [a.version]


def test_champion_light_leaves_the_artifact_in_the_database(db: Session) -> None:
    saved = registry.save_model(db, TARGET, _result("heavy"), promote=True)
    db.commit()
    db.expunge_all()  # a fresh identity map, as an API request has
    light = registry.champion_light(db, TARGET)
    assert light is not None and light.id == saved.id
    assert "artifact" in inspect(light).unloaded  # the ~1.3 MB column was not read
    assert light.version == saved.version and light.metrics["brier"] == 0.64
    assert light.feature_names == FEATURES and light.params["n_estimators"] == 100
    assert light.artifact == _result("heavy").artifact  # still there on demand
    assert registry.champion_light(db, "NIFTY BANK") is None
    db.expunge_all()
    full = registry.champion(db, TARGET)
    assert full is not None and "artifact" not in inspect(full).unloaded


# ------------------------------------------------------------------ promotion rule


def test_window_brier_is_n_weighted_and_bounded() -> None:
    folds = _folds([0.70, 0.60, 0.50])  # test_end 2026-10-30; folds are 30 days each
    assert folds[-1].test_end == TRAINED_THROUGH and folds[0].test_start == date(2026, 8, 2)
    whole = registry.window_brier(folds, date(2026, 1, 1), TRAINED_THROUGH)
    assert whole == (pytest.approx(0.6), 63)
    # Only folds tested entirely inside [start, end] count; the boundaries are inclusive.
    assert registry.window_brier(folds, folds[1].test_start, TRAINED_THROUGH) == (0.55, 42)
    assert registry.window_brier(
        folds, folds[1].test_start + timedelta(days=1), TRAINED_THROUGH
    ) == (
        0.5,
        21,
    )
    assert registry.window_brier(folds, date(2026, 1, 1), TRAINED_THROUGH - timedelta(days=1)) == (
        pytest.approx(0.65),
        42,
    )
    assert registry.window_brier(folds, date(2027, 1, 1), date(2027, 6, 1)) is None
    # n-weighting, and the stored dict form (ISO dates) reads the same as the dataclasses.
    heavy = _folds([0.80], end=date(2026, 7, 31), n_test=63) + _folds([0.50])
    assert registry.window_brier(heavy, date(2026, 1, 1), TRAINED_THROUGH) == (
        pytest.approx((0.8 * 63 + 0.5 * 21) / 84),
        84,
    )
    as_dicts = _metrics(0.6, 0.4, heavy).as_dict()["folds"]
    assert isinstance(as_dicts, list)
    assert registry.window_brier(as_dicts, date(2026, 1, 1), TRAINED_THROUGH) == (
        pytest.approx((0.8 * 63 + 0.5 * 21) / 84),
        84,
    )
    # Folds without a usable date, size or Brier are skipped, never guessed.
    broken = [
        *as_dicts,
        {"test_start": "nope", "test_end": "2026-10-30", "n_test": 5, "brier": 0.1},
    ]
    broken.append({"test_start": "2026-10-01", "test_end": "2026-10-30", "n_test": 0, "brier": 0.1})
    broken.append(
        {"test_start": "2026-10-01", "test_end": "2026-10-30", "n_test": 9, "brier": None}
    )
    assert registry.window_brier(broken, date(2026, 1, 1), TRAINED_THROUGH) == (
        pytest.approx((0.8 * 63 + 0.5 * 21) / 84),
        84,
    )


def test_no_champion_promotes() -> None:
    ok, why = registry.should_promote(None, _metrics(0.70, 0.40))
    assert ok and "no champion" in why


def test_promotion_is_decided_on_the_common_recent_window() -> None:
    """Twelve 30-day folds back from 30 Oct 2026: the last six lie inside the 182-day window
    (from 1 May 2026), the first six before it."""
    gain = get_settings().retrain_min_brier_gain  # 0.005 by default
    champ_folds = _folds([0.66] * 6 + [0.650] * 6)
    champ = _metrics(0.655, 0.45, champ_folds).as_dict()  # how the champion is stored
    assert champ_folds[6].test_start >= WINDOW_START > champ_folds[5].test_start

    # Better inside the window by the gain, much worse before it (and worse overall).
    beat = _metrics(0.700, 0.40, _folds([0.80] * 6 + [0.650 - gain] * 6))
    ok, why = registry.should_promote(champ, beat)
    assert ok and why.startswith("promote (beat the champion on the common window)")
    assert f"{WINDOW_START.isoformat()}..{TRAINED_THROUGH.isoformat()}" in why
    assert "challenger Brier 0.6450 (n=126) vs champion 0.6500 (n=126)" in why
    assert "needs <= 0.6450" in why

    # Better overall and before the window, equal inside it: not promoted.
    outside_only = _metrics(0.600, 0.60, _folds([0.50] * 6 + [0.650] * 6))
    ok, why = registry.should_promote(champ, outside_only)
    assert not ok and why.startswith("reject (Brier gain too small on the common window")
    assert "data refresh not applicable: champion row not given" in why
    assert "challenger Brier 0.6500 (n=126) vs champion 0.6500 (n=126)" in why

    # A fold that straddles the window start, or the champion's end, is left out on both sides.
    shifted = _folds([0.1] * 6 + [0.650 - gain] * 6, end=TRAINED_THROUGH + timedelta(days=10))
    ok, why = registry.should_promote(champ, _metrics(0.6, 0.5, shifted))
    assert ok and "challenger Brier 0.6450 (n=105)" in why  # 5 folds fit, the last one does not

    # Just short of the gain: rejected; the configured gain is what counts.
    close = _metrics(0.60, 0.5, _folds([0.1] * 6 + [0.650 - gain + 0.001] * 6))
    assert not registry.should_promote(champ, close)[0]
    looser = get_settings().model_copy(update={"retrain_min_brier_gain": 0.003})
    assert registry.should_promote(champ, close, settings=looser)[0]


def test_data_refresh_promotes_the_same_recipe_on_more_rows() -> None:
    gain = get_settings().retrain_min_brier_gain
    champ_result = _result("champ", 0.655, 0.45, _folds([0.66] * 6 + [0.650] * 6), train_rows=1200)
    champ = _champion_row(champ_result)
    # Same features and recipe, a different out-of-sample choice, 21 more rows, a fold past
    # the champion's end (excluded), and a window Brier a hair worse: a data refresh.
    refreshed = _result(
        "fresh",
        0.70,
        0.40,
        _folds([0.66] * 6 + [0.652] * 6 + [0.9], end=TRAINED_THROUGH + timedelta(days=30)),
        params={**RECIPE, "n_estimators": 175, "shrink": 0.6},
        train_rows=1221,
        trained_through=TRAINED_THROUGH + timedelta(days=30),
    )
    ok, why = registry.should_promote(champ, refreshed)
    assert ok and why.startswith("promote (data refresh: same features and params, 1221 vs 1200")
    assert "challenger Brier 0.6520 (n=126) vs champion 0.6500 (n=126)" in why

    def rejected(**changes: Any) -> str:
        fields = {
            "tag": "x",
            "brier": 0.70,
            "accuracy": 0.40,
            "folds": refreshed.metrics.folds,
            "params": refreshed.params,
            "train_rows": 1221,
            "trained_through": refreshed.trained_through,
        }
        fields.update(changes)
        ok, why = registry.should_promote(champ, _result(**fields))
        assert not ok and why.startswith("reject (Brier gain too small on the common window")
        return why

    assert "not more training rows (1200 vs 1200)" in rejected(train_rows=1200)
    assert "different feature set" in rejected(feature_names=["ret_1d"])
    assert "different params" in rejected(params={**refreshed.params, "num_leaves": 7})
    worse = _folds([0.66] * 6 + [0.650 + gain + 0.001] * 6)
    assert f"worse by more than {gain:.4f}" in rejected(folds=worse)
    # Only the challenger's metrics (no TrainResult): the recipe cannot be checked.
    ok, why = registry.should_promote(champ, refreshed.metrics)
    assert not ok and "challenger TrainResult not given" in why
    # Beating the champion still promotes regardless of the recipe.
    ok, why = registry.should_promote(
        champ, _result("new", 0.7, 0.4, _folds([0.9] * 6 + [0.640] * 6), feature_names=["ret_1d"])
    )
    assert ok and why.startswith("promote (beat the champion")
    # The champion's trained_through, not its folds, ends the window when the row is given.
    late_champ = _champion_row(
        _result("c2", 0.655, 0.45, champ_result.metrics.folds, trained_through=date(2026, 12, 31))
    )
    ok, why = registry.should_promote(late_champ, refreshed)
    assert "2026-07-02..2026-12-31" in why and "(n=84)" in why  # 4 champion folds remain inside


def test_whole_sample_rule_when_there_is_no_common_window() -> None:
    s = get_settings()
    gain = s.retrain_min_brier_gain
    champ = {"brier": 0.650, "accuracy": 0.500}  # a champion stored without fold metrics
    ok, why = registry.should_promote(champ, _metrics(0.650 - gain, 0.495))
    assert ok and why.startswith("promote (whole-sample rule)") and "0.6450" in why
    assert "no common window (champion folds: 0, challenger folds: 0)" in why
    ok, why = registry.should_promote(champ, _metrics(0.650 - gain + 0.001, 0.52))
    assert not ok and "Brier gain too small" in why
    ok, why = registry.should_promote(champ, _metrics(0.600, 0.485))
    assert not ok and "accuracy fell" in why
    ok, why = registry.should_promote(champ, _metrics(0.600, 0.490))
    assert ok  # exactly one point lower is allowed

    tighter = s.model_copy(update={"retrain_min_brier_gain": 0.05})
    assert not registry.should_promote(champ, _metrics(0.620, 0.52), settings=tighter)[0]
    assert registry.should_promote(champ, _metrics(0.600, 0.52), settings=tighter)[0]

    # Champion folds exist but none in the window (e.g. a champion whose history ended
    # long before): the whole-sample rule, with the window attempt reported.
    old = _metrics(0.650, 0.5, _folds([0.6] * 3, end=date(2025, 1, 31))).as_dict()
    ok, why = registry.should_promote(old, _metrics(0.600, 0.5, _folds([0.5] * 3)))
    assert ok and "champion folds: 3, challenger folds: 3" in why

    ok, why = registry.should_promote({"accuracy": 0.5}, _metrics(0.60, 0.5))
    assert ok and "no comparable" in why
    ok, why = registry.should_promote(champ, _metrics(float("nan"), 0.5))
    assert not ok and "challenger has no" in why


# ------------------------------------------------------------------ MLflow (optional)


class _FakeMlflow(types.ModuleType):
    def __init__(self, fail: bool = False) -> None:
        super().__init__("mlflow")
        self.calls: list[tuple[str, Any]] = []
        self.fail = fail

    def set_tracking_uri(self, uri: str) -> None:
        if self.fail:
            raise ConnectionError("tracking server down")
        self.calls.append(("uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("experiment", name))

    @contextmanager
    def start_run(self, run_name: str) -> Any:
        self.calls.append(("run", run_name))
        yield None

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("tags", tags))

    def log_params(self, params: dict[str, str]) -> None:
        self.calls.append(("params", params))

    def log_metrics(self, m: dict[str, float]) -> None:
        self.calls.append(("metrics", m))

    def log_dict(self, d: dict[str, object], name: str) -> None:
        self.calls.append(("dict", name))

    def log_text(self, s: str, name: str) -> None:
        self.calls.append(("text", name))


def test_log_to_mlflow_is_a_noop_without_a_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMlflow(fail=True)  # would raise if touched
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    s = get_settings().model_copy(update={"mlflow_tracking_uri": ""})
    registry.log_to_mlflow(_result("a"), TARGET, promoted=True, settings=s)
    assert fake.calls == []


def test_log_to_mlflow_logs_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    s = get_settings().model_copy(update={"mlflow_tracking_uri": "http://mlflow.local:5000"})
    registry.log_to_mlflow(
        _result("a", brier=0.61, accuracy=0.53), TARGET, promoted=False, settings=s
    )
    kinds = [k for k, _ in fake.calls]
    assert kinds[:3] == ["uri", "experiment", "run"]
    assert {"tags", "params", "metrics", "dict", "text"} <= set(kinds)
    metrics_logged = dict(fake.calls)["metrics"]
    assert metrics_logged["brier"] == 0.61 and metrics_logged["accuracy"] == 0.53
    assert metrics_logged["baseline_always_up_accuracy"] == 0.4
    assert all(isinstance(v, float) for v in metrics_logged.values())
    assert dict(fake.calls)["tags"]["promoted"] == "false"

    broken = _FakeMlflow(fail=True)
    monkeypatch.setitem(sys.modules, "mlflow", broken)
    registry.log_to_mlflow(_result("a"), TARGET, promoted=True, settings=s)  # must not raise
