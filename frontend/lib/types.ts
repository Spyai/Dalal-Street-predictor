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

export interface Track {
  n_graded: number;
  accuracy: number | null;
  best_baseline: string | null;
  best_baseline_label: string | null;
  best_baseline_accuracy: number | null;
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
}

export interface Today {
  clock: MarketClock;
  cues: Cue[];
  flows: Flows;
  kite: KiteStatus;
  market_call: MarketCall[];
  market_call_note: string;
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
  n: number;
  accuracy: number | null;
}

export interface OosMetrics {
  n_oos: number | null;
  accuracy: number | null;
  brier: number | null;
  log_loss: number | null;
}

export interface Champion {
  id: number;
  version: string;
  kind: string;
  trained_through: string;
  train_rows: number;
  oos: OosMetrics;
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
