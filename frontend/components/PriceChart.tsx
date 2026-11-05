"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import type { CandleRow, Level, SeriesPoint } from "@/lib/types";

export interface Overlays {
  sma20: boolean;
  sma50: boolean;
  sma200: boolean;
  bollinger: boolean;
  levels: boolean;
  rsi: boolean;
  macd: boolean;
}

interface Props {
  candles: CandleRow[];
  indicators: Record<string, SeriesPoint[]>;
  levels: Level[];
  overlays: Overlays;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const toPoints = (pts: SeriesPoint[] | undefined) =>
  (pts ?? []).map((p) => ({ time: p.time as Time, value: p.value }));

export default function PriceChart({ candles, indicators, levels, overlays }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || candles.length === 0) return;
    let chart: IChartApi | null = null;

    const build = () => {
      chart?.remove();
      const c = {
        text: cssVar("--muted"),
        grid: cssVar("--grid"),
        border: cssVar("--border"),
        up: cssVar("--up"),
        down: cssVar("--down"),
        accent: cssVar("--accent"),
        warn: cssVar("--warn"),
        faint: cssVar("--faint"),
      };
      chart = createChart(el, {
        autoSize: true,
        layout: {
          background: { type: ColorType.Solid, color: "transparent" },
          textColor: c.text,
          fontSize: 11,
          panes: { separatorColor: c.border, separatorHoverColor: c.border },
          attributionLogo: true,
        },
        grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
        rightPriceScale: { borderColor: c.border },
        timeScale: { borderColor: c.border },
        crosshair: { mode: CrosshairMode.Normal },
      });

      const price = chart.addSeries(CandlestickSeries, {
        upColor: c.up,
        downColor: c.down,
        borderVisible: false,
        wickUpColor: c.up,
        wickDownColor: c.down,
      });
      price.setData(
        candles.map((k) => ({
          time: k.time as Time,
          open: k.open,
          high: k.high,
          low: k.low,
          close: k.close,
        })),
      );

      const vol = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "vol",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      vol.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      vol.setData(
        candles
          .filter((k) => k.volume !== null)
          .map((k) => ({
            time: k.time as Time,
            value: k.volume as number,
            color: k.close >= k.open ? `${c.up}55` : `${c.down}55`,
          })),
      );

      const line = (key: string, color: string, width: 1 | 2 = 1, style = LineStyle.Solid) => {
        const s = chart!.addSeries(LineSeries, {
          color,
          lineWidth: width,
          lineStyle: style,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        s.setData(toPoints(indicators[key]));
      };
      if (overlays.sma20) line("sma20", "#e0b84f");
      if (overlays.sma50) line("sma50", "#5fb3f6");
      if (overlays.sma200) line("sma200", "#c084fc", 2);
      if (overlays.bollinger) {
        line("bb_upper", c.faint, 1, LineStyle.Dotted);
        line("bb_mid", c.faint, 1, LineStyle.Dotted);
        line("bb_lower", c.faint, 1, LineStyle.Dotted);
      }
      if (overlays.levels) {
        for (const lv of levels) {
          price.createPriceLine({
            price: lv.price,
            color: lv.kind === "support" ? c.up : c.down,
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `${lv.kind === "support" ? "S" : "R"} ×${lv.touches}`,
          });
        }
      }

      let pane = 1;
      if (overlays.rsi) {
        const rsi = chart.addSeries(
          LineSeries,
          { color: c.accent, lineWidth: 1, priceLineVisible: false },
          pane,
        );
        rsi.setData(toPoints(indicators.rsi14));
        rsi.createPriceLine({ price: 70, color: c.faint, lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: false, title: "" });
        rsi.createPriceLine({ price: 30, color: c.faint, lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: false, title: "" });
        chart.panes()[pane]?.setHeight(90);
        pane += 1;
      }
      if (overlays.macd) {
        const hist = chart.addSeries(
          HistogramSeries,
          { priceLineVisible: false, lastValueVisible: false },
          pane,
        );
        hist.setData(
          (indicators.macd_hist ?? []).map((p) => ({
            time: p.time as Time,
            value: p.value,
            color: p.value >= 0 ? `${c.up}88` : `${c.down}88`,
          })),
        );
        const m = chart.addSeries(LineSeries, { color: c.accent, lineWidth: 1, priceLineVisible: false }, pane);
        m.setData(toPoints(indicators.macd));
        const s = chart.addSeries(LineSeries, { color: c.warn, lineWidth: 1, priceLineVisible: false }, pane);
        s.setData(toPoints(indicators.macd_signal));
        chart.panes()[pane]?.setHeight(90);
      }
      chart.timeScale().fitContent();
    };

    build();
    window.addEventListener("dalal-theme", build);
    return () => {
      window.removeEventListener("dalal-theme", build);
      chart?.remove();
    };
  }, [candles, indicators, levels, overlays]);

  return <div ref={ref} className="h-[460px] w-full sm:h-[540px]" />;
}
