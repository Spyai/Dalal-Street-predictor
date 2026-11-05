"""Pydantic schemas for all API input/output."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------- auth


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    totp: str = Field(min_length=6, max_length=8)


class MeOut(BaseModel):
    username: str


class SetupStatusOut(BaseModel):
    needs_setup: bool
    setup_enabled: bool


class SetupIn(BaseModel):
    token: str = Field(min_length=1, max_length=256)
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=12, max_length=256)


class SetupOut(BaseModel):
    username: str
    totp_secret: str
    provisioning_uri: str
    qr_svg_data_url: str


# ----------------------------------------------------------------------------- today


class MarketClockOut(BaseModel):
    now_ist: datetime
    session: str
    is_trading_day: bool
    day_reason: str | None
    calendar_verified: bool
    next_trading_day: date
    previous_trading_day: date


class CueOut(BaseModel):
    key: str
    label: str
    group: str
    unit: str
    last: float | None
    prev_close: float | None
    change_pct: float | None
    change_bps: float | None
    as_of: datetime | None
    status: str
    source: str | None
    note: str | None


class FlowDayOut(BaseModel):
    trade_date: date
    fii_net_cr: float | None
    dii_net_cr: float | None
    fii_buy_cr: float | None
    fii_sell_cr: float | None
    dii_buy_cr: float | None
    dii_sell_cr: float | None
    provisional: bool
    source: str


class FlowsOut(BaseModel):
    days: list[FlowDayOut]
    status: str  # ok | stale | unavailable
    expected_latest: date
    note: str | None


class KiteStatusOut(BaseModel):
    configured: bool
    connected: bool
    kite_user_id: str | None
    expires_at: datetime | None
    redirect_url: str


# ----------------------------------------------------------------------------- market call


class DriverOut(BaseModel):
    """One plain-English reason behind a call (app.ml.core.Driver)."""

    feature: str
    value: float | None
    contribution: float | None
    text: str


class GradeOut(BaseModel):
    """What the market did, written after the fact. All None until graded."""

    actual_open: float | None
    actual_close: float | None
    actual_pct: float | None
    actual_class: str | None
    correct: bool | None
    brier: float | None
    log_loss: float | None
    gap_actual_pct: float | None
    gap_correct: bool | None
    graded_at: datetime | None


class TrackOut(BaseModel):
    """The live track record that sits beside every call (honesty rules, brief §1).

    The baseline comparison is paired: `best_baseline_accuracy` and `model_paired_accuracy`
    cover exactly the `paired_n` graded sessions where that baseline had a call.
    `comparison` is one of app.services.scorecard.COMPARISONS (ahead, behind, level,
    too_early, no_baseline, no_calls); ahead/behind needs `min_paired_n` paired calls.
    """

    n_graded: int
    accuracy: float | None
    best_baseline: str | None
    best_baseline_label: str | None
    best_baseline_accuracy: float | None
    paired_n: int | None
    model_paired_accuracy: float | None
    min_paired_n: int
    comparison: str
    sample_warning: bool
    min_sample_size: int


class MarketCallOut(BaseModel):
    """The ledger's current call for one target and session, plus grade and track record."""

    id: int
    seq: int
    target: str
    session_date: date
    as_of: datetime
    is_late: bool
    reference_date: date
    reference_close: float
    predicted_class: str
    probs: dict[str, float]
    range_p10_pct: float | None
    range_p90_pct: float | None
    gap_call: str | None
    gap_expected_pct: float | None
    confidence: str
    no_edge: bool
    drivers: list[DriverOut]
    timing_guidance: str | None
    baselines: dict[str, str | None]
    model_version: str | None
    note: str | None
    grade: GradeOut | None
    track: TrackOut | None
    # True when the upcoming session has no call yet and this is the latest earlier call
    # (with its grade), shown so the close grade is visible after 15:30 IST.
    is_previous: bool = False


class TodayOut(BaseModel):
    clock: MarketClockOut
    cues: list[CueOut]
    flows: FlowsOut
    kite: KiteStatusOut
    market_call_session: date  # the session the page is for (core.session_for)
    market_call: list[MarketCallOut] = Field(default_factory=list)
    market_call_note: str = ""  # why a target has no call at all; empty when every one has
    ideas: TodayIdeasOut  # the session's stock ideas per horizon ("No edge today" when none)


# ----------------------------------------------------------------------------- scorecard


class ConfidenceBucketOut(BaseModel):
    confidence: str
    n: int
    accuracy: float | None = None  # the service omits the key when n == 0


class CalibrationBinOut(BaseModel):
    bin_lo: float
    bin_hi: float
    n: int
    predicted_mean: float | None
    observed: float | None


class BaselineScoreOut(BaseModel):
    """A baseline against the model on the `n` graded sessions where the baseline had a call."""

    name: str
    label: str
    n: int
    accuracy: float | None  # the baseline's, on those n sessions
    model_accuracy: float | None  # the model's, on the same n sessions


class OosMetricsOut(BaseModel):
    """The champion's walk-forward, out-of-sample metrics."""

    n_oos: int | None
    accuracy: float | None
    brier: float | None
    log_loss: float | None


class OosBaselineOut(BaseModel):
    """A baseline's walk-forward, out-of-sample score stored with the champion."""

    name: str
    label: str
    n: int
    accuracy: float | None
    brier: float | None


class ChampionOut(BaseModel):
    id: int
    version: str
    kind: str
    trained_through: date
    train_rows: int
    oos: OosMetricsOut
    baselines: list[OosBaselineOut]  # empty when the model was stored without them
    class_frequencies: dict[str, float | None] | None
    range_coverage: float | None  # share of OOS moves inside the 10–90 % range, if stored
    climatology_brier: float | None  # OOS Brier of the class-frequency forecast
    oos_verdict: str
    promoted_at: datetime | None
    created_at: datetime | None


class PredictionRowOut(BaseModel):
    """One ledger record with its grade (grade fields None until graded)."""

    id: int
    seq: int
    target: str
    session_date: date
    as_of: datetime | None
    is_late: bool
    predicted_class: str
    probs: dict[str, float]
    range_p10_pct: float | None
    range_p90_pct: float | None
    confidence: str
    no_edge: bool
    gap_call: str | None
    gap_expected_pct: float | None
    model_version_id: int | None
    corrects_prediction_id: int | None
    actual_open: float | None
    actual_close: float | None
    actual_pct: float | None
    actual_class: str | None
    correct: bool | None
    brier: float | None
    log_loss: float | None
    gap_actual_pct: float | None
    gap_correct: bool | None
    graded_at: datetime | None


class ScorecardOut(BaseModel):
    """Mirrors app.services.scorecard.Scorecard, plus the targets the switch can offer."""

    target: str
    targets: list[str]
    n_graded: int
    accuracy: float | None
    brier: float | None
    log_loss: float | None
    gap_n: int
    gap_accuracy: float | None
    by_confidence: list[ConfidenceBucketOut]
    calibration: list[CalibrationBinOut]
    baselines: list[BaselineScoreOut]
    best_baseline: BaselineScoreOut | None  # the toughest paired comparison
    comparison: str  # app.services.scorecard.COMPARISONS
    min_paired_n: int
    verdict: str
    sample_warning: bool
    min_sample_size: int
    champion: ChampionOut | None
    recent: list[PredictionRowOut]


class LedgerVerifyOut(BaseModel):
    ok: bool
    checked: int
    first_bad_seq: int | None
    message: str


# ----------------------------------------------------------------------------- stocks


class InstrumentOut(BaseModel):
    symbol: str
    exchange: str
    name: str | None
    industry: str | None
    in_universe: bool
    is_index: bool


class CandleOut(BaseModel):
    time: date
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    delivery_pct: float | None


class SeriesPoint(BaseModel):
    time: date
    value: float


class LevelOut(BaseModel):
    price: float
    touches: int
    kind: str


class StockOut(BaseModel):
    instrument: InstrumentOut
    candles: list[CandleOut]
    indicators: dict[str, list[SeriesPoint]]
    levels: list[LevelOut]
    last_close: float | None
    prev_close: float | None
    change_pct: float | None
    data_as_of: date | None
    data_status: str  # ok | stale | missing
    sources: list[str]
    notes: list[str]


# ----------------------------------------------------------------------------- health


class JobHealthOut(BaseModel):
    name: str
    label: str
    schedule: str
    phase: int
    built: bool
    last_status: str | None
    last_run_at: datetime | None
    last_finished_at: datetime | None
    last_message: str | None
    last_success_at: datetime | None


class SourceHealthOut(BaseModel):
    source: str
    label: str
    status: str
    last_success_at: datetime | None
    last_error_at: datetime | None
    last_error: str | None
    data_as_of: datetime | None
    hint: str | None


class FreshnessOut(BaseModel):
    dataset: str
    latest: date | datetime | None
    expected: date | None
    status: str
    detail: str | None = None


class ModelHealthOut(BaseModel):
    """A champion model in production and its out-of-sample metrics."""

    name: str
    target: str
    version: str
    trained_through: date
    oos_brier: float | None
    climatology_brier: float | None  # OOS Brier of the class-frequency forecast, same sessions
    oos_accuracy: float | None
    n_oos: int | None
    promoted_at: datetime | None


class HealthOut(BaseModel):
    now: datetime
    scheduler_mode: str
    scheduler_alive: bool
    scheduler_heartbeat: datetime | None
    database: str
    redis: str
    cache_backend: str
    jobs: list[JobHealthOut]
    sources: list[SourceHealthOut]
    freshness: list[FreshnessOut]
    models: list[ModelHealthOut]
    models_note: str
    holiday_coverage: dict[int, int]


# ----------------------------------------------------------------------------- factors


class FactorOut(BaseModel):
    """One feature behind a call or an idea: its value in the recorded snapshot and the
    model's contribution to the prediction (None when the ledger holds no contribution for
    it; nothing is estimated after the fact). `text` is the recorded driver sentence when
    the factor was one of the stored drivers."""

    feature: str
    label: str
    value: float | None
    contribution: float | None
    text: str | None


class FactorsOut(BaseModel):
    """The full factor list behind one ledger record, for detail views: the top
    `drivers_top_n` drivers, then every feature in the snapshot with its contribution."""

    drivers: list[DriverOut]
    all_factors: list[FactorOut]  # drivers first (largest |contribution| first), then the rest
    feature_snapshot: dict[str, float | None]
    n_features: int
    n_with_contribution: int
    drivers_top_n: int


# ----------------------------------------------------------------------------- ideas


class IdeaTrackOut(BaseModel):
    """The live record of this setup at this horizon, shown beside every idea's p_target
    (app.services.ideas_scorecard.IdeaTrack)."""

    horizon: str
    setup: str | None
    n_graded: int
    hit_rate: float | None  # share of graded ideas whose target came before the stop
    win_rate: float | None  # share with a positive net return after costs
    expectancy_pct: float | None
    sample_warning: bool
    min_sample_size: int


class IdeaOutcomeOut(BaseModel):
    """What happened to an idea (idea_outcome). Status 'pending' with every field None
    until the 09:20 fill; graded fields None until the exit."""

    status: str
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    gross_return_pct: float | None
    costs_pct: float | None
    net_return_pct: float | None
    r_multiple: float | None
    days_held: int | None
    max_favorable_pct: float | None
    max_adverse_pct: float | None
    graded_at: datetime | None
    source: str | None
    notes: str | None
    error_category: str | None


class PaperTradeOut(BaseModel):
    """One simulated position (app.services.paper.trade_dict). Never a real order."""

    id: int
    idea_id: int | None
    symbol: str
    exchange: str
    side: str
    qty: int
    entry_date: date
    entry_price: float
    entry_value: float | None
    stop: float | None
    target: float | None
    exit_date: date | None
    exit_price: float | None
    status: str
    gross_pnl: float | None
    costs: float | None
    net_pnl: float | None
    net_return_pct: float | None
    r_multiple: float | None
    note: str | None
    created_at: datetime | None
    closed_at: datetime | None


class IdeaOut(BaseModel):
    """One current ledger idea as the lists show it: levels, p_target with its live track
    record, EV after costs, position, outcome. Surveillance is never checked (no ASM/GSM
    source is reachable), so `surveillance_unknown` is True and `surveillance_note` says so."""

    id: int
    seq: int
    session_date: date
    as_of: datetime
    is_late: bool
    instrument_id: int
    symbol: str
    exchange: str
    horizon: str
    horizon_days: int
    setup: str
    setup_label: str
    strategy: str
    strategy_label: str
    direction: str
    entry_low: float
    entry_high: float
    stop: float
    target_1: float
    target_2: float | None
    p_target: float
    expected_return_low_pct: float
    expected_return_high_pct: float
    risk_reward: float
    ev_pct_after_costs: float
    position_qty: int
    position_value: float
    risk_amount: float
    rank: int
    surveillance_unknown: bool
    surveillance_note: str
    model_version_id: int | None
    model_version: str | None
    corrects_idea_id: int | None
    corrected_by_seq: int | None  # set when a later record corrects this one (superseded)
    note: str | None
    outcome: IdeaOutcomeOut
    track: IdeaTrackOut
    # True when shown on Today as the latest earlier session's idea (no idea for the
    # upcoming session yet), so its outcome stays visible after the close.
    is_previous: bool = False


class IdeaDetailOut(IdeaOut):
    """An idea with its full reasoning: reasons, invalidation, every factor, paper trade."""

    reasons: list[str]
    invalidation: str
    factors: FactorsOut
    paper_trade: PaperTradeOut | None
    hash: str
    prev_hash: str


class IdeasListOut(BaseModel):
    ideas: list[IdeaOut]
    n: int
    status: str | None
    horizon: str | None
    setup: str | None
    limit: int
    statuses: list[str]  # accepted status filters
    horizons: list[str]
    setups: list[str]  # the known setups (strategies' plus any in the ledger)
    min_sample_size: int
    surveillance_note: str


class HorizonIdeasOut(BaseModel):
    """The Today page's ideas for one horizon. `no_edge` is True only when the engine ran
    for the session (a champion exists and the session's market call is in the ledger)
    and recorded no idea; `note` says in words why the list is empty otherwise."""

    horizon: str
    horizon_days: int
    target_atr: float
    stop_atr: float
    ideas: list[IdeaOut]
    no_edge: bool
    note: str | None
    champion_version: str | None


class TodayIdeasOut(BaseModel):
    session: date  # the session the ideas are for (core.session_for, or the previous one)
    is_previous: bool  # no idea for the upcoming session yet; these are the latest earlier ones
    by_horizon: list[HorizonIdeasOut]
    n_ideas: int
    note: str  # empty when every horizon has an idea
    surveillance_note: str
    min_p_target: float
    min_ev_pct: float
    max_per_horizon: int


# ----------------------------------------------------------------------------- paper


class EquityPointOut(BaseModel):
    trade_date: date
    equity: float
    cash: float
    invested: float
    open_positions: int
    nifty_close: float | None  # Nifty 50 price index (TRI unavailable)
    nifty_rebased: float | None  # rebased to the paper capital at the first known close


class PaperOut(BaseModel):
    """Mirrors app.services.paper.PortfolioSummary plus the settings behind it. Every
    metric is None when there is nothing to compute it from; n_* are the sample sizes."""

    capital: float
    n_trades: int  # closed trades behind the trade statistics
    n_open: int
    n_wins: int
    n_losses: int
    hit_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy: float | None
    expectancy_pct: float | None
    profit_factor: float | None
    total_net_pnl: float | None
    n_days: int  # equity snapshots
    n_returns: int  # daily returns behind Sharpe / Sortino
    max_drawdown_pct: float | None
    sharpe: float | None
    sortino: float | None
    total_return_pct: float | None
    nifty_return_pct: float | None
    min_cash: float | None
    max_exposure_pct: float | None
    levered_days: int
    equity_curve: list[EquityPointOut]
    open_trades: list[PaperTradeOut]
    recent_closed: list[PaperTradeOut]
    sample_warning: bool
    note: str
    min_sample_size: int
    risk_per_trade_pct: float
    max_position_pct: float
    cost_round_trip_pct: float


# ----------------------------------------------------------------------------- ideas scorecard


class IdeaGroupStatsOut(BaseModel):
    """group_stats for one setup or horizon (both horizons are always listed)."""

    setup: str | None = None
    horizon: str | None = None
    n: int
    hit_rate: float | None
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy_pct: float | None
    profit_factor: float | None
    avg_r_multiple: float | None
    avg_days_held: float | None


class IdeaOosOut(BaseModel):
    """An ideas champion's walk-forward, out-of-sample metrics (None when not stored)."""

    n_oos: int | None
    hit_rate: float | None  # of the candidates that would have cleared the p/EV bar
    n_selected: int | None
    hit_rate_ranked: float | None
    n_ranked: int | None
    brier: float | None
    log_loss: float | None
    accuracy: float | None
    expectancy_after_costs_pct: float | None
    expectancy_ranked_pct: float | None
    expectancy_all_candidates_pct: float | None
    cost_round_trip_pct: float | None
    n_sessions: int | None
    embargo_sessions: int | None


class IdeaOosBaselineOut(BaseModel):
    name: str
    label: str
    n: int
    hit_rate: float | None
    brier: float | None


class IdeaChampionOut(BaseModel):
    id: int
    name: str
    horizon: str
    version: str
    kind: str
    trained_through: date
    train_rows: int
    oos: IdeaOosOut
    baselines: list[IdeaOosBaselineOut]
    per_strategy: dict[str, dict[str, float | int | None]]
    calibration: list[CalibrationBinOut]
    class_frequencies: dict[str, float | None] | None
    notes: list[str]
    oos_verdict: str
    promoted_at: datetime | None
    created_at: datetime | None


class IdeaRowOut(BaseModel):
    """One ledger idea with its outcome, exactly as app.services.ideas_scorecard.row_dict
    returns it (outcome fields None while pending; status 'pending' without an outcome row)."""

    id: int
    seq: int
    session_date: date
    as_of: datetime | None
    is_late: bool
    instrument_id: int
    symbol: str
    exchange: str
    horizon: str
    horizon_days: int
    setup: str
    strategy: str
    direction: str
    entry_low: float
    entry_high: float
    stop: float
    target_1: float
    target_2: float | None
    p_target: float
    expected_return_low_pct: float
    expected_return_high_pct: float
    risk_reward: float
    ev_pct_after_costs: float
    position_qty: int
    position_value: float
    risk_amount: float
    rank: int
    surveillance_unknown: bool
    model_version_id: int | None
    corrects_idea_id: int | None
    note: str | None
    status: str
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    gross_return_pct: float | None
    costs_pct: float | None
    net_return_pct: float | None
    r_multiple: float | None
    days_held: int | None
    max_favorable_pct: float | None
    max_adverse_pct: float | None
    graded_at: datetime | None
    outcome_source: str | None
    outcome_notes: str | None
    error_category: str | None


class IdeasScorecardOut(BaseModel):
    """Mirrors app.services.ideas_scorecard.IdeasScorecard, plus the horizons the switch
    offers and the sample-size threshold."""

    horizon: str | None
    horizons: list[str]
    n_ideas: int
    n_graded: int
    by_status: dict[str, int]
    hit_rate: float | None
    win_rate: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy_pct: float | None
    profit_factor: float | None
    avg_r_multiple: float | None
    avg_days_held: float | None
    by_setup: list[IdeaGroupStatsOut]
    by_horizon: list[IdeaGroupStatsOut]
    calibration: list[CalibrationBinOut]
    sample_warning: bool
    min_sample_size: int
    verdict: str
    champions: dict[str, IdeaChampionOut | None]
    recent: list[IdeaRowOut]
