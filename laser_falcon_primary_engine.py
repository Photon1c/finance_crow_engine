"""Laser Falcon primary orchestration — CSV replay options research engine.

No live trading. No broker APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from anomaly_detection_engine import detect_anomalies
from chain_integrity_engine import assess_chain_integrity, json_safe_integrity, validate_chain_for_analysis
from laser_falcon_data_adapter import (
    DEFAULT_OPTION_DIR,
    DEFAULT_STOCK_DIR,
    LaserFalconSnapshot,
    load_laser_falcon_snapshot,
)
from laser_falcon_regime_mapper import map_laser_falcon_regime_metrics
from options_pressure_mapper import compute_options_pressure_metrics
from ou_iv_engine import plot_ou_iv_projection, simulate_ou_iv_paths
from projection_range_engine import DEFAULT_PROJECTION_DAYS, clamp_projection_days
from regime_detection_engine import classify_vol_regime
from stochastic_vol_engine import plot_stochastic_vol_projection, simulate_stochastic_vol_paths
from temporal_chain_differential_engine import compare_ticker_chain_dates
from volatility_arbitrage_detector import detect_vol_arbitrage
from volatility_skew_engine import (
    compare_skew_to_benchmark,
    compute_skew_metrics,
    plot_iv_skew,
)
from volatility_surface_engine import build_iv_surface_grid, plot_iv_surface

DEFAULT_OUTPUT_DIR = Path("outputs/laser_falcon")
DEFAULT_BENCHMARK = "SPY"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_skew_report(
    snapshot: LaserFalconSnapshot,
    *,
    expiration: Optional[str] = None,
    benchmark_snapshot: Optional[LaserFalconSnapshot] = None,
) -> dict[str, Any]:
    metrics = compute_skew_metrics(snapshot.option_df, spot=snapshot.spot, expiration=expiration)
    report: dict[str, Any] = {"ticker": snapshot.ticker, "skew": metrics}
    if benchmark_snapshot is not None:
        bench = compute_skew_metrics(benchmark_snapshot.option_df, spot=benchmark_snapshot.spot, expiration=expiration)
        report["benchmark"] = compare_skew_to_benchmark(metrics, bench)
    return report


def build_surface_report(snapshot: LaserFalconSnapshot) -> dict[str, Any]:
    return build_iv_surface_grid(snapshot.option_df, spot=snapshot.spot, ticker=snapshot.ticker)


def run_ou_iv_projection(
    *,
    iv0: float,
    long_run_mean_iv: float = 0.25,
    theta: float = 4.0,
    vol_of_vol: float = 0.15,
    projection_days: int = DEFAULT_PROJECTION_DAYS,
    n_paths: int = 500,
    seed: Optional[int] = 42,
) -> dict[str, Any]:
    result = simulate_ou_iv_paths(
        iv0=iv0,
        long_run_mean_iv=long_run_mean_iv,
        theta=theta,
        vol_of_vol=vol_of_vol,
        projection_days=projection_days,
        n_paths=n_paths,
        seed=seed,
    )
    export = {k: v for k, v in result.items() if k != "paths"}
    export["paths_sample"] = result["paths"][:5].tolist()
    return export


def run_stochastic_vol_projection(
    *,
    spot0: float,
    atm_iv: float,
    projection_days: int = DEFAULT_PROJECTION_DAYS,
    n_paths: int = 500,
    rho: float = -0.7,
    xi: float = 0.5,
    kappa: float = 2.0,
    seed: Optional[int] = 42,
) -> dict[str, Any]:
    variance0 = max(atm_iv ** 2, 1e-6)
    result = simulate_stochastic_vol_paths(
        spot0=spot0,
        variance0=variance0,
        theta=variance0,
        kappa=kappa,
        xi=xi,
        rho=rho,
        projection_days=projection_days,
        n_paths=n_paths,
        seed=seed,
    )
    export = {k: v for k, v in result.items() if k not in ("prices", "variances", "vol_paths")}
    return export


def _write_summary_md(
    path: Path,
    *,
    ticker: str,
    benchmark: str,
    snapshot: LaserFalconSnapshot,
    skew_report: dict[str, Any],
    surface_report: dict[str, Any],
    ou_result: dict[str, Any],
    stoch_result: dict[str, Any],
    regime_metrics: dict[str, Any],
    pressure_metrics: dict[str, Any],
    anomaly: dict[str, Any],
    vol_regime: dict[str, Any],
    vol_arbitrage: dict[str, Any],
    temporal_diff: dict[str, Any],
    chain_integrity: dict[str, Any],
    artifact_paths: dict[str, str],
) -> None:
    skew = skew_report.get("skew", {})
    lines = [
        f"# Laser Falcon Summary — {ticker.upper()}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Data Health",
        f"- Status: **{snapshot.data_health.get('status', 'UNKNOWN')}**",
        f"- Contracts: {snapshot.data_health.get('n_contracts', 0)}",
        f"- Expirations: {snapshot.data_health.get('n_expirations', 0)}",
        f"- IV coverage: {snapshot.data_health.get('iv_coverage_pct', 0)}%",
        "",
        "## Chain Integrity",
        f"- Status: **{chain_integrity.get('status', 'n/a')}**",
        f"- Health score: {chain_integrity.get('chain_health_score', 'n/a')}",
        f"- Blank expirations: {chain_integrity.get('blank_expiration_count', 0)}",
        f"- Missing IV ratio: {chain_integrity.get('missing_iv_ratio', 'n/a')}",
        f"- Wide spread ratio: {chain_integrity.get('wide_spread_ratio', 'n/a')}",
        f"- Duplicate contracts: {chain_integrity.get('duplicate_contract_count', 0)}",
        "",
    ]
    for warning in chain_integrity.get("warnings", []):
        lines.append(f"- Chain warning: {warning}")
    for warning in snapshot.data_health.get("warnings", []):
        lines.append(f"- Warning: {warning}")
    lines.extend(
        [
            "",
            "## IV Skew",
            f"- Expiration: {skew.get('expiration', 'n/a')}",
            f"- ATM IV: {skew.get('atm_iv', 'n/a')}",
            f"- Put wing IV: {skew.get('put_wing_iv', 'n/a')}",
            f"- Call wing IV: {skew.get('call_wing_iv', 'n/a')}",
            f"- Skew slope: {skew.get('skew_slope', 'n/a')}",
            f"- Skew ratio: {skew.get('skew_ratio', 'n/a')}",
            f"- Skew asymmetry: {skew.get('skew_asymmetry_pressure', 'n/a')}",
            f"- Calls overpriced: {skew.get('calls_overpriced_flag', False)}",
            f"- Puts overpriced: {skew.get('puts_overpriced_flag', False)}",
            f"- Skew inversion: {skew.get('skew_inversion_flag', False)}",
            "",
            "## Options Pressure Metrics",
            f"- Gamma compression: {pressure_metrics.get('gamma_compression_score', 'n/a')}",
            f"- Vol expansion (ATM/realized): {pressure_metrics.get('volatility_expansion_score', 'n/a')}",
            f"- Skew asymmetry pressure: {pressure_metrics.get('skew_asymmetry_pressure', 'n/a')}",
            f"- Dealer hedging stress: {pressure_metrics.get('dealer_hedging_stress_score', 'n/a')}",
            f"- 30d realized vol: {pressure_metrics.get('realized_vol_30d_pct', 'n/a')}%",
            "",
            "## Anomaly Detection",
            f"- Primary: **{anomaly.get('primary_label', 'n/a')}**",
            f"- Labels: {', '.join(anomaly.get('labels', []))}",
            f"- Severity: {anomaly.get('severity_score', 'n/a')}",
            "",
            "## Vol Regime",
            f"- Regime: **{vol_regime.get('regime', 'n/a')}**",
            f"- Confidence: {vol_regime.get('confidence', 'n/a')}",
            "",
            "## Vol Arbitrage / Dislocation",
            f"- Status: {vol_arbitrage.get('status', 'n/a')}",
            f"- Dislocation: {vol_arbitrage.get('dislocation_pct', 'n/a')}",
            f"- Potential dislocation: {vol_arbitrage.get('potential_dislocation', False)}",
            "",
            "## Temporal Chain Differential",
            f"- Status: {temporal_diff.get('status', temporal_diff.get('pressure_direction', 'n/a'))}",
            f"- Prior date: {temporal_diff.get('prior_chain_date', 'n/a')}",
            f"- Current date: {temporal_diff.get('current_chain_date', 'n/a')}",
            f"- Pressure direction: **{temporal_diff.get('pressure_direction', 'n/a')}**",
            f"- delta ATM IV: {(temporal_diff.get('deltas') or {}).get('delta_atm_iv', 'n/a')}",
            f"- delta call wing IV: {(temporal_diff.get('deltas') or {}).get('delta_call_wing_iv', 'n/a')}",
            f"- delta put wing IV: {(temporal_diff.get('deltas') or {}).get('delta_put_wing_iv', 'n/a')}",
            f"- delta dealer stress: {(temporal_diff.get('deltas') or {}).get('delta_dealer_stress', 'n/a')}",
            f"- delta skew asymmetry: {(temporal_diff.get('deltas') or {}).get('delta_skew', 'n/a')}",
            f"- Compatibility: {((temporal_diff.get('compatibility') or {}).get('status', 'n/a'))}",
            f"- Contract universe drift: {((temporal_diff.get('compatibility') or {}).get('contract_universe_drift_flag', False))}",
            "",
            "## IV Surface",
            f"- Status: {surface_report.get('status', 'n/a')}",
            f"- Reason: {surface_report.get('density', {}).get('reason', 'n/a')}",
            "",
            "## OU IV Mean Reversion",
            f"- IV0: {ou_result.get('iv0', 'n/a')}",
            f"- Terminal mean: {ou_result.get('terminal_mean', 'n/a')}",
            f"- Half-life (days): {ou_result.get('half_life_days', 'n/a')}",
            "",
            "## Stochastic Volatility",
            f"- Terminal median price: {stoch_result.get('terminal_price_p50', 'n/a')}",
            f"- Terminal median vol: {stoch_result.get('terminal_vol_p50', 'n/a')}",
            "",
            "## Pressure-Field Mapping (local)",
            f"- iv_pressure_score: {regime_metrics.get('iv_pressure_score')}",
            f"- skew_instability_score: {regime_metrics.get('skew_instability_score')}",
            f"- surface_dislocation_score: {regime_metrics.get('surface_dislocation_score')}",
            f"- vol_reversion_pressure: {regime_metrics.get('vol_reversion_pressure')}",
            f"- option_liquidity_risk: {regime_metrics.get('option_liquidity_risk')}",
            f"- energy_injection_proxy (E_i): {regime_metrics.get('energy_injection_proxy', 'n/a')}",
            f"- boundary_stress_proxy (B_s): {regime_metrics.get('boundary_stress_proxy', 'n/a')}",
            f"- rupture_pressure_contributor: {regime_metrics.get('rupture_pressure_contributor', 'n/a')}",
            f"- lrp_contributor: {regime_metrics.get('lrp_contributor', 'n/a')}",
            "",
            f"## Benchmark: {benchmark.upper()}",
        ]
    )
    bench = skew_report.get("benchmark")
    if bench:
        lines.append(f"- ATM IV delta vs {benchmark.upper()}: {bench.get('atm_iv_delta', 'n/a')}")
    else:
        lines.append("- Benchmark comparison unavailable")
    lines.extend(["", "## Artifacts"] + [f"- {k}: `{v}`" for k, v in artifact_paths.items()])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_laser_falcon_analysis(
    ticker: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    projection_days: int = DEFAULT_PROJECTION_DAYS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stock_dir: str = DEFAULT_STOCK_DIR,
    option_dir: str = DEFAULT_OPTION_DIR,
    ou_theta: float = 4.0,
    ou_long_run_iv: Optional[float] = None,
    vol_of_vol: float = 0.15,
    stoch_rho: float = -0.7,
    stoch_xi: float = 0.5,
    n_paths: int = 500,
) -> dict[str, Any]:
    """Run full Laser Falcon pipeline and save artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker = ticker.upper()
    benchmark = benchmark.upper()
    projection_days = clamp_projection_days(projection_days)

    snapshot = load_laser_falcon_snapshot(ticker, stock_dir=stock_dir, option_dir=option_dir)
    chain_integrity = json_safe_integrity(
        assess_chain_integrity(snapshot.option_df, ticker=ticker, spot_price=snapshot.spot)
    )
    bench_snapshot: Optional[LaserFalconSnapshot] = None
    if benchmark != ticker:
        try:
            bench_snapshot = load_laser_falcon_snapshot(benchmark, stock_dir=stock_dir, option_dir=option_dir)
        except FileNotFoundError:
            bench_snapshot = None

    skew_report = build_skew_report(snapshot, benchmark_snapshot=bench_snapshot)
    bench_metrics = None
    if bench_snapshot is not None:
        bench_metrics = compute_skew_metrics(bench_snapshot.option_df, spot=bench_snapshot.spot)

    skew_path = plot_iv_skew(
        snapshot.option_df,
        ticker=ticker,
        spot=snapshot.spot,
        output_path=output_dir / f"{ticker}_iv_skew.png",
        benchmark_metrics=bench_metrics,
    )
    surface_path, surface_report = plot_iv_surface(
        snapshot.option_df,
        ticker=ticker,
        spot=snapshot.spot,
        output_path=output_dir / f"{ticker}_iv_surface.png",
    )

    atm_iv = skew_report["skew"].get("atm_iv") or 25.0
    iv0 = float(atm_iv) / 100.0 if atm_iv > 1 else float(atm_iv)
    long_run = ou_long_run_iv if ou_long_run_iv is not None else max(iv0 * 0.85, 0.15)

    ou_full = simulate_ou_iv_paths(
        iv0=iv0,
        long_run_mean_iv=long_run,
        theta=ou_theta,
        vol_of_vol=vol_of_vol,
        projection_days=projection_days,
        n_paths=n_paths,
    )
    ou_path = plot_ou_iv_projection(ou_full, ticker=ticker, output_path=output_dir / f"{ticker}_ou_iv_projection.png")
    ou_export = run_ou_iv_projection(
        iv0=iv0,
        long_run_mean_iv=long_run,
        theta=ou_theta,
        vol_of_vol=vol_of_vol,
        projection_days=projection_days,
        n_paths=n_paths,
    )

    stoch_full = simulate_stochastic_vol_paths(
        spot0=snapshot.spot,
        variance0=iv0 ** 2,
        theta=iv0 ** 2,
        xi=stoch_xi,
        rho=stoch_rho,
        projection_days=projection_days,
        n_paths=n_paths,
    )
    stoch_path = plot_stochastic_vol_projection(
        stoch_full,
        ticker=ticker,
        output_path=output_dir / f"{ticker}_stochastic_vol_projection.png",
    )
    stoch_export = run_stochastic_vol_projection(
        spot0=snapshot.spot,
        atm_iv=iv0,
        projection_days=projection_days,
        n_paths=n_paths,
        rho=stoch_rho,
        xi=stoch_xi,
    )

    temporal_diff = compare_ticker_chain_dates(
        ticker,
        stock_dir=stock_dir,
        option_dir=option_dir,
    )

    pressure_metrics = compute_options_pressure_metrics(
        option_df=snapshot.option_df,
        stock_df=snapshot.stock_df,
        spot=snapshot.spot,
        skew_metrics=skew_report["skew"],
        chain_integrity=chain_integrity,
    )
    bench_skew = bench_metrics if bench_metrics else None
    anomaly = detect_anomalies(
        skew_metrics=skew_report["skew"],
        pressure_metrics=pressure_metrics,
        benchmark_skew=bench_skew,
        data_health=snapshot.data_health,
        chain_integrity=chain_integrity,
        compatibility=temporal_diff.get("compatibility"),
    )
    vol_regime = classify_vol_regime(
        skew_metrics=skew_report["skew"],
        pressure_metrics=pressure_metrics,
        ou_result=ou_export,
        anomaly=anomaly,
    )
    vol_arbitrage: dict[str, Any] = {"status": "SKIPPED"}
    if bench_snapshot is not None:
        vol_arbitrage = detect_vol_arbitrage(
            snapshot.option_df,
            bench_snapshot.option_df,
            target_spot=snapshot.spot,
            benchmark_spot=bench_snapshot.spot,
            target_ticker=ticker,
            benchmark_ticker=benchmark,
        )

    regime_metrics = map_laser_falcon_regime_metrics(
        skew_metrics=skew_report["skew"],
        surface_report=surface_report,
        ou_result=ou_export,
        data_health=snapshot.data_health,
        pressure_metrics=pressure_metrics,
    )

    artifact_paths = {
        "iv_skew": str(skew_path),
        "iv_surface": str(surface_path),
        "ou_iv_projection": str(ou_path),
        "stochastic_vol_projection": str(stoch_path),
    }

    summary_path = output_dir / f"{ticker}_laser_falcon_summary.md"
    _write_summary_md(
        summary_path,
        ticker=ticker,
        benchmark=benchmark,
        snapshot=snapshot,
        skew_report=skew_report,
        surface_report=surface_report,
        ou_result=ou_export,
        stoch_result=stoch_export,
        regime_metrics=regime_metrics,
        pressure_metrics=pressure_metrics,
        anomaly=anomaly,
        vol_regime=vol_regime,
        vol_arbitrage=vol_arbitrage,
        temporal_diff=temporal_diff,
        chain_integrity=chain_integrity,
        artifact_paths=artifact_paths,
    )

    latest = {
        "ticker": ticker,
        "benchmark": benchmark,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spot": snapshot.spot,
        "chain_date": snapshot.chain_date,
        "data_health": snapshot.data_health,
        "chain_integrity": chain_integrity,
        "skew": skew_report,
        "surface": {k: v for k, v in surface_report.items() if k != "grid"},
        "ou_iv": ou_export,
        "stochastic_vol": stoch_export,
        "pressure_metrics": pressure_metrics,
        "anomaly": anomaly,
        "vol_regime": vol_regime,
        "vol_arbitrage": vol_arbitrage,
        "temporal_diff": temporal_diff,
        "regime_metrics": regime_metrics,
        "projection_days": projection_days,
        "artifacts": artifact_paths,
        "summary_md": str(summary_path),
    }
    json_path = output_dir / f"{ticker}_laser_falcon_latest.json"
    json_path.write_text(json.dumps(_json_safe(latest), indent=2), encoding="utf-8")
    latest["json_path"] = str(json_path)
    return latest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Laser Falcon options research engine (CSV replay only)")
    parser.add_argument("--ticker", default="SPCX", help="Target ticker")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark ticker")
    parser.add_argument("--projection-days", type=int, default=DEFAULT_PROJECTION_DAYS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stock-dir", default=DEFAULT_STOCK_DIR)
    parser.add_argument("--option-dir", default=DEFAULT_OPTION_DIR)
    parser.add_argument("--ou-theta", type=float, default=4.0)
    parser.add_argument("--vol-of-vol", type=float, default=0.15)
    parser.add_argument("--stoch-rho", type=float, default=-0.7)
    parser.add_argument("--n-paths", type=int, default=500)
    args = parser.parse_args(argv)

    try:
        result = run_laser_falcon_analysis(
            args.ticker,
            benchmark=args.benchmark,
            projection_days=args.projection_days,
            output_dir=Path(args.output_dir),
            stock_dir=args.stock_dir,
            option_dir=args.option_dir,
            ou_theta=args.ou_theta,
            vol_of_vol=args.vol_of_vol,
            stoch_rho=args.stoch_rho,
            n_paths=args.n_paths,
        )
    except FileNotFoundError as exc:
        print(f"Laser Falcon error: {exc}", file=sys.stderr)
        return 1

    print(f"Laser Falcon complete for {result['ticker']}")
    print(f"  Summary: {result['summary_md']}")
    print(f"  JSON:    {result['json_path']}")
    for name, path in result["artifacts"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
