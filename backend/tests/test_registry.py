"""Model registry: save / champion / promote with the real database, promotion rules, MLflow."""

from __future__ import annotations

import re
import sys
import types
from contextlib import contextmanager
from datetime import date
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml import registry
from app.ml.core import TrainResult, WalkForwardMetrics
from app.models import ModelVersion

TARGET = "NIFTY 50"


def _metrics(brier: float, accuracy: float) -> WalkForwardMetrics:
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
    )


def _result(tag: str, brier: float = 0.64, accuracy: float = 0.5) -> TrainResult:
    return TrainResult(
        artifact=f'{{"classifier": "{tag}", "q10": "", "q90": ""}}',
        feature_names=["ret_1d", "es_fut_ret_1d"],
        params={"seed": 42, "n_estimators": 300},
        metrics=_metrics(brier, accuracy),
        trained_through=date(2026, 10, 30),
        train_rows=1200,
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


# ------------------------------------------------------------------ promotion rule


def test_should_promote_rules() -> None:
    s = get_settings()
    gain = s.retrain_min_brier_gain  # 0.005 by default
    ok, why = registry.should_promote(None, _metrics(0.70, 0.40))
    assert ok and "no champion" in why

    champ = {"brier": 0.650, "accuracy": 0.500}
    ok, why = registry.should_promote(champ, _metrics(0.650 - gain, 0.495))
    assert ok and why.startswith("promote") and "0.6450" in why
    ok, why = registry.should_promote(champ, _metrics(0.650 - gain + 0.001, 0.52))
    assert not ok and "Brier gain too small" in why
    ok, why = registry.should_promote(champ, _metrics(0.600, 0.485))
    assert not ok and "accuracy fell" in why
    ok, why = registry.should_promote(champ, _metrics(0.600, 0.490))
    assert ok  # exactly one point lower is allowed

    tighter = s.model_copy(update={"retrain_min_brier_gain": 0.05})
    assert not registry.should_promote(champ, _metrics(0.620, 0.52), settings=tighter)[0]
    assert registry.should_promote(champ, _metrics(0.600, 0.52), settings=tighter)[0]

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
