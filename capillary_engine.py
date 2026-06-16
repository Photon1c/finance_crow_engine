"""Capillary Engine — microscopic noise absorption vs persistence during cruise mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

DEFAULT_TICKER = "SPY"
DEFAULT_INPUT_TEMPLATE = "outputs/canopyento_boundary_{ticker}.csv"
DEFAULT_OUTPUT_CSV_TEMPLATE = "outputs/capillary_engine_{ticker}.csv"
DEFAULT_OUTPUT_MD_TEMPLATE = "outputs/capillary_engine_{ticker}.md"
DEFAULT_OUTPUT_JSON_TEMPLATE = "outputs/capillary_engine_latest_{ticker}.json"

OUTPUT_COLUMNS = [
    "date",
    "close",
    "returns",
    "brownian_noise",
    "wave_persistence",
    "compression",
    "surface_tension",
    "capillary_score",
    "capillary_regime",
    "cruise_integrity",
    "pinch_off_risk",
    "canopyento_regime",
    "canopyento_score",
    "canopyento_rupture_prob",
    "combined_read",
]

CAPILLARY_REGIMES = (
    "ABSORBING_NOISE",
    "CRUISE_SURFACE_ACTIVE",
    "SURFACE_RIPPLING",
    "PINCH_OFF_WATCH",
    "CAPILLARY_RUPTURE_RISK",
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize_series_01(series: pd.Series, *, floor: float = 0.0, ceiling: Optional[float] = None) -> pd.Series:
    """Map a series into [0, 1] using robust clipping."""
    if ceiling is None:
        valid = series.dropna()
        ceiling = float(valid.quantile(0.95)) if len(valid) else 1.0
    if ceiling <= floor:
        ceiling = floor + 1.0
    return ((series - floor) / (ceiling - floor)).clip(0.0, 1.0)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the latest observation within each rolling window."""
    min_periods = max(2, window // 2)

    def rank_last(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.5
        current = values[-1]
        if np.isnan(current):
            return np.nan
        prior = values[:-1]
        prior = prior[~np.isnan(prior)]
        if len(prior) == 0:
            return 0.5
        return float((prior <= current).mean())

    return series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True)


def _rolling_autocorrelation(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """Rolling lag-1 autocorrelation with safe min_periods."""
    min_periods = max(lag + 2, window // 2)

    def autocorr(values: np.ndarray) -> float:
        if len(values) <= lag:
            return np.nan
        left = values[:-lag]
        right = values[lag:]
        mask = ~(np.isnan(left) | np.isnan(right))
        left = left[mask]
        right = right[mask]
        if len(left) < 2:
            return np.nan
        if left.std() == 0 or right.std() == 0:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    return series.rolling(window, min_periods=min_periods).apply(autocorr, raw=True)


def _resolve_column(df: pd.DataFrame, *names: str) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def load_canopyento_csv(path: Path) -> pd.DataFrame:
    """Load and normalize the CanopyEnto boundary output CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CanopyEnto input not found: {path}")

    df = pd.read_csv(path)
    date_col = _resolve_column(df, "Date", "date")
    close_col = _resolve_column(df, "Close", "close")
    if date_col is None or close_col is None:
        raise ValueError("Input CSV must include Date/date and Close/close columns.")

    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    rename_map = {}
    for target, aliases in {
        "open": ("Open", "open"),
        "high": ("High", "high"),
        "low": ("Low", "low"),
        "volume": ("Volume", "volume"),
        "B_s": ("B_s", "b_s"),
        "gate_stance": ("gate_stance",),
        "regime_label": ("regime_label", "canopyento_regime"),
        "rupture_pressure_score": ("rupture_pressure_score", "canopyento_score"),
        "rupture_probability": ("rupture_probability", "canopyento_rupture_prob"),
        "stance_quadrant": ("stance_quadrant",),
    }.items():
        source = _resolve_column(df, *aliases)
        if source is not None:
            rename_map[source] = target

    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["date", "close"]).sort_values("date", ascending=True)
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable rows found in {path}")

    return df


def compute_capillary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute capillary instability metrics from CanopyEnto-enriched OHLCV data."""
    result = df.copy()
    close = result["close"]
    returns = close.pct_change()
    result["returns"] = returns

    # --- brownian_noise: short-window chop relative to longer baseline ---
    short_std = returns.rolling(5, min_periods=3).std()
    long_std = returns.rolling(20, min_periods=8).std()
    brownian_raw = short_std / long_std.replace(0, np.nan)
    result["brownian_noise"] = _normalize_series_01(brownian_raw, floor=0.0, ceiling=2.0)

    # --- wave_persistence: do absolute-return disturbances echo or dissipate? ---
    abs_returns = returns.abs()
    wave_raw = _rolling_autocorrelation(abs_returns, window=10, lag=1)
    # Autocorrelation lives on [-1, 1]; map to [0, 1] where persistence is high.
    result["wave_persistence"] = ((wave_raw + 1.0) / 2.0).clip(0.0, 1.0)

    # --- compression: tighter daily range => higher compression ---
    if {"high", "low"}.issubset(result.columns):
        daily_range = (pd.to_numeric(result["high"], errors="coerce") - pd.to_numeric(result["low"], errors="coerce")) / close
    else:
        # Close-only fallback when high/low are unavailable.
        daily_range = returns.rolling(5, min_periods=3).std()

    range_percentile = _rolling_percentile_rank(daily_range, window=20)
    result["compression"] = (1.0 - range_percentile).clip(0.0, 1.0)

    # --- surface_tension: absorption / stabilizing force ---
    realized_vol = returns.rolling(20, min_periods=8).std()
    inverse_realized_vol = 1.0 / (1.0 + realized_vol * np.sqrt(252.0) * 10.0)

    ma20 = close.rolling(20, min_periods=8).mean()
    distance_from_ma = (close - ma20).abs() / close.replace(0, np.nan)
    inverse_distance_from_ma = (1.0 - distance_from_ma * 20.0).clip(0.0, 1.0)

    if {"high", "low", "volume"}.issubset(result.columns):
        typical_price = (
            pd.to_numeric(result["high"], errors="coerce")
            + pd.to_numeric(result["low"], errors="coerce")
            + close
        ) / 3.0
        volume = pd.to_numeric(result["volume"], errors="coerce")
        vol_sum = volume.rolling(20, min_periods=8).sum()
        vwap = (typical_price * volume).rolling(20, min_periods=8).sum() / vol_sum.replace(0, np.nan)
        distance_from_vwap = (close - vwap).abs() / close.replace(0, np.nan)
        inverse_distance_from_vwap = (1.0 - distance_from_vwap * 20.0).clip(0.0, 1.0)
    else:
        inverse_distance_from_vwap = inverse_distance_from_ma

    cruise_gate_bonus = pd.Series(0.0, index=result.index)
    if "gate_stance" in result.columns:
        cruise_gate_bonus = (
            result["gate_stance"]
            .astype(str)
            .str.contains("CRUISE", case=False, na=False)
            .astype(float)
            * 0.25
        )

    b_s_stability = pd.Series(0.5, index=result.index)
    if "B_s" in result.columns:
        b_s = pd.to_numeric(result["B_s"], errors="coerce")
        b_s_delta = b_s.diff().abs().rolling(5, min_periods=2).mean()
        b_s_stability = (1.0 - b_s_delta * 4.0).clip(0.0, 1.0)
        moderate_b_s = (1.0 - (b_s - 0.20).abs() * 2.0).clip(0.0, 1.0)
        b_s_stability = (b_s_stability + moderate_b_s) / 2.0

    surface_parts = pd.concat(
        [
            inverse_realized_vol,
            inverse_distance_from_ma,
            inverse_distance_from_vwap,
            cruise_gate_bonus,
            b_s_stability,
        ],
        axis=1,
    )
    result["surface_tension"] = surface_parts.mean(axis=1, skipna=True).clip(0.0, 1.0)

    # --- capillary_score: Brownian noise × persistence × compression ÷ surface tension ---
    numerator = result["brownian_noise"] * result["wave_persistence"] * result["compression"]
    capillary_raw = numerator / result["surface_tension"].clip(lower=0.05)
    capillary_ceiling = float(capillary_raw.dropna().quantile(0.95)) if capillary_raw.notna().any() else 1.0
    if capillary_ceiling <= 0:
        capillary_ceiling = 1.0
    result["capillary_score"] = (capillary_raw / capillary_ceiling).clip(0.0, 1.0)

    result["capillary_regime"] = result["capillary_score"].map(classify_capillary_regime)
    result["cruise_integrity"] = (result["surface_tension"] * (1.0 - result["capillary_score"])).clip(0.0, 1.0)
    result["pinch_off_risk"] = pd.concat(
        [
            result["capillary_score"],
            result["wave_persistence"],
            result["compression"],
            1.0 - result["surface_tension"],
        ],
        axis=1,
    ).mean(axis=1, skipna=True).clip(0.0, 1.0)

    # CanopyEnto passthrough fields for integration.
    result["canopyento_regime"] = result.get("regime_label", pd.Series(pd.NA, index=result.index))
    result["canopyento_score"] = pd.to_numeric(
        result.get("rupture_pressure_score", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    result["canopyento_rupture_prob"] = pd.to_numeric(
        result.get("rupture_probability", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )

    result["combined_read"] = result.apply(build_combined_read, axis=1)
    return result


def classify_capillary_regime(score: float) -> str:
    """Map capillary score into named instability regimes."""
    if pd.isna(score):
        return ""
    if score < 0.25:
        return "ABSORBING_NOISE"
    if score < 0.45:
        return "CRUISE_SURFACE_ACTIVE"
    if score < 0.65:
        return "SURFACE_RIPPLING"
    if score < 0.80:
        return "PINCH_OFF_WATCH"
    return "CAPILLARY_RUPTURE_RISK"


def build_combined_read(row: pd.Series) -> str:
    """Human-readable synthesis of CanopyEnto and Capillary states."""
    cap_regime = str(row.get("capillary_regime", "") or "")
    canopy_regime = str(row.get("canopyento_regime", "") or "")
    gate = str(row.get("gate_stance", "") or "")
    stance = str(row.get("stance_quadrant", "") or "")

    if gate == "LOW-CONFIDENCE CRUISE MODE" and cap_regime == "ABSORBING_NOISE":
        return (
            "Market remains in cruise mode. Microscopic disturbances are being absorbed."
        )
    if canopy_regime == "PRESSURE_BUILDING" and cap_regime == "SURFACE_RIPPLING":
        return (
            "Stored pressure is rising and micro-noise is persisting. Watch for instability."
        )
    if canopy_regime == "PRESSURE_BUILDING" and cap_regime == "PINCH_OFF_WATCH":
        return (
            "CanopyEnto pressure and Capillary persistence agree. Rupture risk is elevated."
        )
    if "unresolved" in stance.lower() and cap_regime == "CAPILLARY_RUPTURE_RISK":
        return (
            "Directional packet remains unresolved, but surface coherence is failing. "
            "Reduce conviction or wait for confirmation."
        )
    if cap_regime == "ABSORBING_NOISE":
        return "Brownian layer is quiet; microscopic noise is dissipating normally."
    if cap_regime == "CRUISE_SURFACE_ACTIVE":
        return "Cruise surface is active but still absorbing small disturbances."
    if cap_regime == "SURFACE_RIPPLING":
        return "Surface rippling detected; micro-disturbances are beginning to persist."
    if cap_regime == "PINCH_OFF_WATCH":
        return "Pinch-off watch: compression and wave persistence are elevated."
    if cap_regime == "CAPILLARY_RUPTURE_RISK":
        return "Capillary rupture risk: microscopic instability may precede macro release."
    return "Capillary and CanopyEnto signals are mixed; monitor for regime handoff."


def export_capillary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the public output schema."""
    export = df.copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    return export[OUTPUT_COLUMNS]


def write_markdown_report(
    df: pd.DataFrame,
    report_path: Path,
    *,
    ticker: str,
    input_path: Path,
) -> None:
    """Write capillary markdown report with CanopyEnto integration."""
    latest = df.iloc[-1]
    latest_date = latest["date"]
    latest_date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)

    lines = [
        f"# Capillary Engine Report — {ticker.upper()}",
        "",
        f"- **Input:** `{input_path}`",
        f"- **As of:** {latest_date_str}",
        "",
        "## Latest Read",
        "",
        str(latest["combined_read"]),
        "",
        f"- **Capillary regime:** {latest['capillary_regime']}",
        f"- **Capillary score:** {latest['capillary_score']:.4f}",
        f"- **Cruise integrity:** {latest['cruise_integrity']:.4f}",
        f"- **Pinch-off risk:** {latest['pinch_off_risk']:.4f}",
        "",
        "## Metric Table",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| close | {latest['close']:.2f} |",
        f"| returns | {latest['returns']:.6f} |",
        f"| brownian_noise | {latest['brownian_noise']:.4f} |",
        f"| wave_persistence | {latest['wave_persistence']:.4f} |",
        f"| compression | {latest['compression']:.4f} |",
        f"| surface_tension | {latest['surface_tension']:.4f} |",
        f"| capillary_score | {latest['capillary_score']:.4f} |",
        f"| cruise_integrity | {latest['cruise_integrity']:.4f} |",
        f"| pinch_off_risk | {latest['pinch_off_risk']:.4f} |",
        "",
        "## Regime Interpretation",
        "",
        "| Capillary Score | Regime | Meaning |",
        "| :--- | :--- | :--- |",
        "| 0.00–0.25 | ABSORBING_NOISE | Microscopic disturbances dissipate cleanly |",
        "| 0.25–0.45 | CRUISE_SURFACE_ACTIVE | Cruise surface intact with normal noise |",
        "| 0.45–0.65 | SURFACE_RIPPLING | Disturbances begin to echo |",
        "| 0.65–0.80 | PINCH_OFF_WATCH | Compression + persistence rising |",
        "| 0.80–1.00 | CAPILLARY_RUPTURE_RISK | Brownian layer failing; pinch-off risk elevated |",
        "",
        f"**Current:** `{latest['capillary_regime']}`",
        "",
        "## CanopyEnto Integration",
        "",
        "Richter / CanopyEnto detects stored pressure and boundary regime. "
        "Capillary Engine detects whether microscopic noise is still being absorbed. "
        "A market can remain in cruise mode while its Brownian layer becomes increasingly unstable.",
        "",
        "| CanopyEnto Field | Value |",
        "| :--- | :--- |",
        f"| regime | {latest.get('canopyento_regime', '')} |",
        f"| rupture score | {latest.get('canopyento_score', float('nan')):.4f} |",
        f"| rupture probability | {latest.get('canopyento_rupture_prob', float('nan')):.4f} |",
        f"| gate stance | {latest.get('gate_stance', '')} |",
        f"| stance quadrant | {latest.get('stance_quadrant', '')} |",
        "",
        "## Notes / Caveats",
        "",
        "- Capillary metrics are derived from local OHLCV and CanopyEnto CSV fields only.",
        "- Short histories use reduced `min_periods`; early rows may be blank.",
        "- `surface_tension` is a first-pass absorption proxy, not order-flow truth.",
        "- High cruise integrity does not guarantee directional trade permission from CanopyEnto.",
        "- Use Capillary read as a micro-stability overlay, not a standalone directional model.",
        "",
    ]

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_latest_json(
    latest: pd.Series,
    json_path: Path,
    *,
    ticker: str,
) -> None:
    """Write dashboard-friendly latest capillary snapshot."""
    latest_date = latest["date"]
    latest_date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date)

    payload: dict[str, Any] = {
        "ticker": ticker.upper(),
        "as_of_date": latest_date_str,
        "combined_read": latest["combined_read"],
        "capillary_regime": latest["capillary_regime"],
        "metrics": {
            "brownian_noise": float(latest["brownian_noise"]),
            "wave_persistence": float(latest["wave_persistence"]),
            "compression": float(latest["compression"]),
            "surface_tension": float(latest["surface_tension"]),
            "capillary_score": float(latest["capillary_score"]),
            "cruise_integrity": float(latest["cruise_integrity"]),
            "pinch_off_risk": float(latest["pinch_off_risk"]),
        },
        "canopyento": {
            "regime": str(latest.get("canopyento_regime", "")),
            "score": float(latest.get("canopyento_score", np.nan))
            if pd.notna(latest.get("canopyento_score"))
            else None,
            "rupture_probability": float(latest.get("canopyento_rupture_prob", np.nan))
            if pd.notna(latest.get("canopyento_rupture_prob"))
            else None,
            "gate_stance": str(latest.get("gate_stance", "")),
            "stance_quadrant": str(latest.get("stance_quadrant", "")),
        },
    }

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capillary Engine — microscopic noise absorption overlay for CanopyEnto."
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help=f"Ticker symbol (default: {DEFAULT_TICKER})")
    parser.add_argument(
        "--input",
        default=None,
        help="CanopyEnto CSV input path (default: outputs/canopyento_boundary_{TICKER}.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Capillary CSV output path (default: outputs/capillary_engine_{TICKER}.csv)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Markdown report path (default: outputs/capillary_engine_{TICKER}.md)",
    )
    parser.add_argument(
        "--json",
        default=None,
        dest="json_output",
        help="Latest JSON output path (default: outputs/capillary_engine_latest_{TICKER}.json)",
    )

    args = parser.parse_args(argv)
    ticker = args.ticker.upper()
    input_path = Path(args.input or DEFAULT_INPUT_TEMPLATE.format(ticker=ticker))
    output_csv = Path(args.output or DEFAULT_OUTPUT_CSV_TEMPLATE.format(ticker=ticker))
    output_md = Path(args.report or DEFAULT_OUTPUT_MD_TEMPLATE.format(ticker=ticker))
    output_json = Path(args.json_output or DEFAULT_OUTPUT_JSON_TEMPLATE.format(ticker=ticker))

    try:
        source_df = load_canopyento_csv(input_path)
        metrics_df = compute_capillary_metrics(source_df)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    export_df = export_capillary_frame(metrics_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_csv, index=False)

    write_markdown_report(metrics_df, output_md, ticker=ticker, input_path=input_path)
    write_latest_json(metrics_df.iloc[-1], output_json, ticker=ticker)

    latest = metrics_df.iloc[-1]
    print(f"Capillary Engine analysis complete for {ticker}")
    print(f"  Input:  {input_path}")
    print(f"  CSV:    {output_csv}")
    print(f"  Report: {output_md}")
    print(f"  JSON:   {output_json}")
    print(
        f"  Latest: {latest['date']} | close={latest['close']:.2f} | "
        f"capillary={latest['capillary_score']:.4f} | regime={latest['capillary_regime']} | "
        f"cruise_integrity={latest['cruise_integrity']:.4f} | pinch_off={latest['pinch_off_risk']:.4f}"
    )
    print(
        f"  CanopyEnto: regime={latest.get('canopyento_regime', '')} | "
        f"gate={latest.get('gate_stance', '')} | rupture_prob={latest.get('canopyento_rupture_prob', float('nan')):.4f}"
    )
    print(f"  Read: {latest['combined_read']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
