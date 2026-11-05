// Mirrors backend/app/schemas.py. Keep in sync.

export interface MarketClock {
  now_ist: string;
  session: "pre_open" | "open" | "closed" | "holiday" | "weekend";
  is_trading_day: boolean;
  day_reason: string | null;
  calendar_verified: boolean;
  next_trading_day: string;
  previous_trading_day: string;
}

export interface Cue {
  key: string;
  label: string;
  group: string;
  unit: "level" | "price" | "fx" | "yield_pct";
  last: number | null;
  prev_close: number | null;
  change_pct: number | null;
  change_bps: number | null;
  as_of: string | null;
  status: "ok" | "stale" | "unavailable";
  source: string | null;
  note: string | null;
}

export interface FlowDay {
  trade_date: string;
  fii_net_cr: number | null;
  dii_net_cr: number | null;
  fii_buy_cr: number | null;
  fii_sell_cr: number | null;
  dii_buy_cr: number | null;
  dii_sell_cr: number | null;
  provisional: boolean;
  source: string;
}

export interface Flows {
  days: FlowDay[];
  status: "ok" | "stale" | "unavailable";
  expected_latest: string;
  note: string | null;
}

export interface KiteStatus {
  configured: boolean;
  connected: boolean;
  kite_user_id: string | null;
  expires_at: string | null;
  redirect_url: string;
}

export type CallClass = "UP" | "FLAT" | "DOWN";
export type Confidence = "high" | "medium" | "low" | "none";
export type GapCall = "gap_up" | "gap_down" | "flat";

export interface Driver {
  feature: string;
  value: number | null;
  contribution: number | null;
  text: string;
}

export interface Grade {
  actual_open: number | null;
  actual_close: number | null;
  actual_pct: number | null;
  actual_class: CallClass | null;
  correct: boolean | null;
  brier: number | null;
  log_loss: number | null;
  gap_actual_pct: number | null;
  gap_correct: boolean | null;
  graded_at: string | null;
}

// Where the model stands against its toughest baseline on paired calls
// (app.services.scorecard.COMPARISONS).
export type Comparison = "ahead" | "behind" | "level" | "too_early" | "no_baseline" | "no_calls";

export interface Track {
  n_graded: number;
  accuracy: number | null;
  best_baseline: string | null;
  best_baseline_label: string | null;
  // Paired: the baseline's and the model's accuracy on the paired_n sessions where the
  // baseline had a call.
  best_baseline_accuracy: number | null;
  paired_n: number | null;
  model_paired_accuracy: number | null;
  min_paired_n: number;
  comparison: Comparison;
  sample_warning: boolean;
  min_sample_size: number;
}

export interface MarketCall {
  id: number;
  seq: number;
  target: string;
  session_date: string;
  as_of: string;
  is_late: boolean;
  reference_date: string;
  reference_close: number;
  predicted_class: CallClass;
  probs: Partial<Record<CallClass, number>>;
  range_p10_pct: number | null;
  range_p90_pct: number | null;
  gap_call: GapCall | null;
  gap_expected_pct: number | null;
  confidence: Confidence;
  no_edge: boolean;
  drivers: Driver[];
  timing_guidance: string | null;
  baselines: Record<string, string | null>;
  model_version: string | null;
  note: string | null;
  grade: Grade | null;
  track: Track | null;
  // The upcoming session has no call yet; this is the latest earlier call with its grade.
  is_previous: boolean;
}

export interface Today {
  clock: MarketClock;
  cues: Cue[];
  flows: Flows;
  kite: KiteStatus;
  market_call_session: string; // the session the page is for
  market_call: MarketCall[];
  market_call_note: string;
  ideas: TodayIdeas; // the session's stock ideas per horizon ("No edge today" when none)
}

// ---------------------------------------------------------------- factors (calls and ideas)

// One feature behind a record: its value in the recorded snapshot and the model's
// contribution (null when the ledger holds none; nothing is estimated after the fact).
export interface Factor {
  feature: string;
  label: string;
  value: number | null;
  contribution: number | null;
  text: string | null; // the recorded driver sentence when the factor was a stored driver
}

export interface Factors {
  drivers: Driver[]; // the top drivers_top_n by |contribution|
  all_factors: Factor[]; // drivers first, then every other snapshot feature
  feature_snapshot: Record<string, number | null>;
  n_features: number;
  n_with_contribution: number;
  drivers_top_n: number;
}

// ---------------------------------------------------------------- ideas

export type Horizon = "swing" | "positional";
export type IdeaStatus =
  | "pending"
  | "open"
  | "target"
  | "stop"
  | "expired"
  | "invalidated"
  | "void";

export interface IdeaTrack {
  horizon: string;
  setup: string | null;
  n_graded: number;
  hit_rate: number | null; // share of graded ideas whose target came before the stop
  win_rate: number | null; // share with a positive net return after costs
  expectancy_pct: number | null;
  sample_warning: boolean;
  min_sample_size: number;
}

export interface IdeaOutcome {
  status: IdeaStatus;
  entry_date: string | null;
  entry_price: number | null;
  exit_date: string | null;
  exit_price: number | null;
  gross_return_pct: number | null;
  costs_pct: number | null;
  net_return_pct: number | null;
  r_multiple: number | null;
  days_held: number | null;
  max_favorable_pct: number | null;
  max_adverse_pct: number | null;
  graded_at: string | null;
  source: string | null;
  notes: string | null;
  error_category: string | null;
}

export interface PaperTrade {
  id: number;
  idea_id: number | null;
  symbol: string;
  exchange: string;
  side: string;
  qty: number;
  entry_date: string;
  entry_price: number;
  entry_value: number | null;
  stop: number | null;
  target: number | null;
  exit_date: string | null;
  exit_price: number | null;
  status: "open" | "closed";
  gross_pnl: number | null;
  costs: number | null;
  net_pnl: number | null;
  net_return_pct: number | null;
  r_multiple: number | null;
  note: string | null;
  created_at: string | null;
  closed_at: string | null;
}

export interface Idea {
  id: number;
  seq: number;
  session_date: string;
  as_of: string;
  is_late: boolean;
  instrument_id: number;
  symbol: string;
  exchange: string;
  horizon: Horizon;
  horizon_days: number;
  setup: string;
  setup_label: string;
  strategy: string;
  strategy_label: string;
  direction: string;
  entry_low: number;
  entry_high: number;
  stop: number;
  target_1: number;
  target_2: number | null;
  p_target: number;
  expected_return_low_pct: number;
  expected_return_high_pct: number;
  risk_reward: number;
  ev_pct_after_costs: number;
  position_qty: number;
  position_value: number;
  risk_amount: number;
  rank: number;
  surveillance_unknown: boolean; // always true: no ASM/GSM source is reachable
  surveillance_note: string;
  model_version_id: number | null;
  model_version: string | null;
  corrects_idea_id: number | null;
  corrected_by_seq: number | null; // set when a later record supersedes this one
  note: string | null;
  outcome: IdeaOutcome;
  track: IdeaTrack;
  is_previous: boolean;
}

export interface IdeaDetail extends Idea {
  reasons: string[];
  invalidation: string;
  factors: Factors;
  paper_trade: PaperTrade | null;
  hash: string;
  prev_hash: string;
}

export interface IdeasList {
  ideas: Idea[];
  n: number;
  status: string | null;
  horizon: string | null;
  setup: string | null;
  limit: number;
  statuses: string[];
  horizons: string[];
  setups: string[];
  min_sample_size: number;
  surveillance_note: string;
}

export interface HorizonIdeas {
  horizon: Horizon;
  horizon_days: number;
  target_atr: number;
  stop_atr: number;
  ideas: Idea[];
  no_edge: boolean; // the engine ran for the session and recorded no idea
  note: string | null; // why the list is empty, in words
  champion_version: string | null;
}

export interface TodayIdeas {
  session: string;
  is_previous: boolean;
  by_horizon: HorizonIdeas[];
  n_ideas: number;
  note: string;
  surveillance_note: string;
  min_p_target: number;
  min_ev_pct: number;
  max_per_horizon: number;
}

// ---------------------------------------------------------------- paper portfolio

export interface EquityPoint {
  trade_date: string;
  equity: number;
  cash: number;
  invested: number;
  open_positions: number;
  nifty_close: number | null; // Nifty 50 price index (TRI unavailable)
  nifty_rebased: number | null; // rebased to the paper capital
}

export interface Paper {
  capital: number;
  n_trades: number; // closed trades behind the trade statistics
  n_open: number;
  n_wins: number;
  n_losses: number;
  hit_rate: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  expectancy: number | null;
  expectancy_pct: number | null;
  profit_factor: number | null;
  total_net_pnl: number | null;
  n_days: number; // equity snapshots
  n_returns: number; // daily returns behind Sharpe / Sortino
  max_drawdown_pct: number | null;
  sharpe: number | null;
  sortino: number | null;
  total_return_pct: number | null;
  nifty_return_pct: number | null;
  min_cash: number | null; // lowest cash on any snapshot; negative means the book borrowed
  max_exposure_pct: number | null; // highest invested / equity, %
  levered_days: number; // snapshots with negative cash
  equity_curve: EquityPoint[];
  open_trades: PaperTrade[];
  recent_closed: PaperTrade[];
  sample_warning: boolean;
  note: string;
  min_sample_size: number;
  risk_per_trade_pct: number;
  max_position_pct: number;
  cost_round_trip_pct: number;
}

// ---------------------------------------------------------------- ideas scorecard

export interface IdeaGroupStats {
  setup?: string | null;
  horizon?: string | null;
  n: number;
  hit_rate: number | null;
  win_rate: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  expectancy_pct: number | null;
  profit_factor: number | null;
  avg_r_multiple: number | null;
  avg_days_held: number | null;
}

export interface IdeaOos {
  n_oos: number | null;
  hit_rate: number | null;
  n_selected: number | null;
  hit_rate_ranked: number | null;
  n_ranked: number | null;
  brier: number | null;
  log_loss: number | null;
  accuracy: number | null;
  expectancy_after_costs_pct: number | null;
  expectancy_ranked_pct: number | null;
  expectancy_all_candidates_pct: number | null;
  hit_rate_top: number | null;
  n_top: number | null;
  expectancy_top_pct: number | null;
  cost_round_trip_pct: number | null;
  n_sessions: number | null;
  embargo_sessions: number | null;
}

// One decile of out-of-sample P(target): 1 = the lowest tenth, 10 = the highest.
export interface IdeaLiftRow {
  decile: number;
  n: number;
  p_mean: number | null;
  hit_rate: number | null;
  expectancy_pct: number | null;
}

export interface IdeaOosBaseline {
  name: string;
  label: string;
  n: number;
  hit_rate: number | null;
  brier: number | null;
}

export interface IdeaChampion {
  id: number;
  name: string;
  horizon: string;
  version: string;
  kind: string;
  trained_through: string;
  train_rows: number;
  oos: IdeaOos;
  baselines: IdeaOosBaseline[];
  per_strategy: Record<string, Record<string, number | null>>;
  calibration: CalibrationBin[];
  lift: IdeaLiftRow[];
  class_frequencies: Record<string, number | null> | null;
  notes: string[];
  oos_verdict: string;
  promoted_at: string | null;
  created_at: string | null;
}

// One ledger idea with its outcome, as app.services.ideas_scorecard.row_dict returns it.
export interface IdeaRow {
  id: number;
  seq: number;
  session_date: string;
  as_of: string | null;
  is_late: boolean;
  instrument_id: number;
  symbol: string;
  exchange: string;
  horizon: Horizon;
  horizon_days: number;
  setup: string;
  strategy: string;
  direction: string;
  entry_low: number;
  entry_high: number;
  stop: number;
  target_1: number;
  target_2: number | null;
  p_target: number;
  expected_return_low_pct: number;
  expected_return_high_pct: number;
  risk_reward: number;
  ev_pct_after_costs: number;
  position_qty: number;
  position_value: number;
  risk_amount: number;
  rank: number;
  surveillance_unknown: boolean;
  model_version_id: number | null;
  corrects_idea_id: number | null;
  note: string | null;
  status: IdeaStatus;
  entry_date: string | null;
  entry_price: number | null;
  exit_date: string | null;
  exit_price: number | null;
  gross_return_pct: number | null;
  costs_pct: number | null;
  net_return_pct: number | null;
  r_multiple: number | null;
  days_held: number | null;
  max_favorable_pct: number | null;
  max_adverse_pct: number | null;
  graded_at: string | null;
  outcome_source: string | null;
  outcome_notes: string | null;
  error_category: string | null;
}

export interface IdeasScorecard {
  horizon: string | null;
  horizons: string[];
  n_ideas: number;
  n_graded: number;
  by_status: Record<string, number>;
  hit_rate: number | null;
  win_rate: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  expectancy_pct: number | null;
  profit_factor: number | null;
  avg_r_multiple: number | null;
  avg_days_held: number | null;
  by_setup: IdeaGroupStats[];
  by_horizon: IdeaGroupStats[];
  calibration: CalibrationBin[];
  sample_warning: boolean;
  min_sample_size: number;
  verdict: string;
  champions: Record<string, IdeaChampion | null>;
  recent: IdeaRow[];
}

export interface ConfidenceBucket {
  confidence: Confidence;
  n: number;
  accuracy: number | null;
}

export interface CalibrationBin {
  bin_lo: number;
  bin_hi: number;
  n: number;
  predicted_mean: number | null;
  observed: number | null;
}

export interface BaselineScore {
  name: string;
  label: string;
  n: number; // graded sessions where this baseline had a call
  accuracy: number | null; // the baseline's, on those n sessions
  model_accuracy: number | null; // the model's, on the same n sessions
}

export interface OosMetrics {
  n_oos: number | null;
  accuracy: number | null;
  brier: number | null;
  log_loss: number | null;
}

export interface OosBaseline {
  name: string;
  label: string;
  n: number;
  accuracy: number | null;
  brier: number | null;
}

export interface Champion {
  id: number;
  version: string;
  kind: string;
  trained_through: string;
  train_rows: number;
  oos: OosMetrics;
  baselines: OosBaseline[]; // empty when the model was stored without them
  class_frequencies: Partial<Record<CallClass, number | null>> | null;
  range_coverage: number | null;
  climatology_brier: number | null;
  oos_verdict: string;
  promoted_at: string | null;
  created_at: string | null;
}

export interface PredictionRow {
  id: number;
  seq: number;
  target: string;
  session_date: string;
  as_of: string | null;
  is_late: boolean;
  predicted_class: CallClass;
  probs: Partial<Record<CallClass, number>>;
  range_p10_pct: number | null;
  range_p90_pct: number | null;
  confidence: Confidence;
  no_edge: boolean;
  gap_call: GapCall | null;
  gap_expected_pct: number | null;
  model_version_id: number | null;
  corrects_prediction_id: number | null;
  actual_open: number | null;
  actual_close: number | null;
  actual_pct: number | null;
  actual_class: CallClass | null;
  correct: boolean | null;
  brier: number | null;
  log_loss: number | null;
  gap_actual_pct: number | null;
  gap_correct: boolean | null;
  graded_at: string | null;
}

export interface Scorecard {
  target: string;
  targets: string[];
  n_graded: number;
  accuracy: number | null;
  brier: number | null;
  log_loss: number | null;
  gap_n: number;
  gap_accuracy: number | null;
  by_confidence: ConfidenceBucket[];
  calibration: CalibrationBin[];
  baselines: BaselineScore[];
  best_baseline: BaselineScore | null; // the toughest paired comparison
  comparison: Comparison;
  min_paired_n: number;
  verdict: string;
  sample_warning: boolean;
  min_sample_size: number;
  champion: Champion | null;
  recent: PredictionRow[];
}

export interface LedgerVerify {
  ok: boolean;
  checked: number;
  first_bad_seq: number | null;
  message: string;
}

export interface InstrumentInfo {
  symbol: string;
  exchange: string;
  name: string | null;
  industry: string | null;
  in_universe: boolean;
  is_index: boolean;
}

export interface CandleRow {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
  delivery_pct: number | null;
}

export interface SeriesPoint {
  time: string;
  value: number;
}

export interface Level {
  price: number;
  touches: number;
  kind: "support" | "resistance";
}

export interface Stock {
  instrument: InstrumentInfo;
  candles: CandleRow[];
  indicators: Record<string, SeriesPoint[]>;
  levels: Level[];
  last_close: number | null;
  prev_close: number | null;
  change_pct: number | null;
  data_as_of: string | null;
  data_status: "ok" | "stale" | "missing";
  sources: string[];
  notes: string[];
}

export interface JobHealth {
  name: string;
  label: string;
  schedule: string;
  phase: number;
  built: boolean;
  last_status: string | null;
  last_run_at: string | null;
  last_finished_at: string | null;
  last_message: string | null;
  last_success_at: string | null;
}

export interface SourceHealth {
  source: string;
  label: string;
  status: string;
  last_success_at: string | null;
  last_error_at: string | null;
  last_error: string | null;
  data_as_of: string | null;
  hint: string | null;
}

export interface Freshness {
  dataset: string;
  latest: string | null;
  expected: string | null;
  status: string;
  detail: string | null;
}

export interface SetupStatus {
  needs_setup: boolean;
  setup_enabled: boolean;
}

export interface SetupResult {
  username: string;
  totp_secret: string;
  provisioning_uri: string;
  qr_svg_data_url: string;
}

export interface ModelHealth {
  name: string;
  target: string;
  version: string;
  trained_through: string;
  oos_brier: number | null;
  climatology_brier: number | null; // OOS Brier of the class-frequency forecast
  oos_accuracy: number | null;
  n_oos: number | null;
  promoted_at: string | null;
}

export interface Health {
  now: string;
  scheduler_mode: "internal" | "external";
  scheduler_alive: boolean;
  scheduler_heartbeat: string | null;
  database: string;
  redis: string;
  cache_backend: "redis" | "postgres";
  jobs: JobHealth[];
  sources: SourceHealth[];
  freshness: Freshness[];
  models: ModelHealth[];
  models_note: string;
  holiday_coverage: Record<string, number>;
}
