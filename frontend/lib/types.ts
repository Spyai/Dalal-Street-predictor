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

export interface Today {
  clock: MarketClock;
  cues: Cue[];
  flows: Flows;
  kite: KiteStatus;
  market_call: null;
  market_call_note: string;
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

export interface Health {
  now: string;
  scheduler_mode: "internal" | "external";
  scheduler_alive: boolean;
  scheduler_heartbeat: string | null;
  database: string;
  redis: string;
  jobs: JobHealth[];
  sources: SourceHealth[];
  freshness: Freshness[];
  models: Record<string, string>[];
  models_note: string;
  holiday_coverage: Record<string, number>;
}
