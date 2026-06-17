"""Streamlit UI for Laser Falcon options research (CSV replay only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from anomaly_detection_engine import detect_anomalies
from chain_integrity_engine import assess_chain_integrity
from laser_falcon_data_adapter import list_expirations, load_laser_falcon_snapshot
from laser_falcon_primary_engine import build_skew_report, run_laser_falcon_analysis, run_ou_iv_projection, run_stochastic_vol_projection
from options_pressure_mapper import compute_options_pressure_metrics
from projection_range_engine import PROJECTION_PRESETS, clamp_projection_days
from regime_detection_engine import classify_vol_regime
from temporal_chain_differential_engine import compare_ticker_chain_dates, list_option_chain_dates
from volatility_skew_engine import compute_skew_metrics, plot_iv_skew
from volatility_surface_engine import plot_iv_surface

st.set_page_config(page_title="Laser Falcon Options Engine", layout="wide")
st.title("Laser Falcon Options Engine")
st.caption("CSV replay and analysis only — no live trading or broker APIs.")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="SPCX").upper()
with col2:
    benchmark = st.text_input("Benchmark", value="SPY").upper()
with col3:
    expiration = "Auto"

st.subheader("Projection Window")
preset_cols = st.columns(len(PROJECTION_PRESETS))
preset_days = PROJECTION_PRESETS[2]
for idx, days in enumerate(PROJECTION_PRESETS):
    if preset_cols[idx].button(f"{days}d", use_container_width=True):
        preset_days = days
projection_days = st.slider(
    "Projection days",
    min_value=1,
    max_value=180,
    value=preset_days,
    key="projection_slider",
)
projection_days = clamp_projection_days(projection_days)

ou_theta = st.slider("OU theta (mean reversion speed)", min_value=0.5, max_value=12.0, value=4.0, step=0.5)
ou_long_run_iv = st.slider("OU long-run IV", min_value=0.05, max_value=1.0, value=0.25, step=0.01)
vol_of_vol = st.slider("Vol-of-vol (OU sigma)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
stoch_rho = st.slider("Stochastic vol rho (price-vol correlation)", min_value=-0.99, max_value=0.99, value=-0.70, step=0.01)
n_paths = st.slider("Simulation paths", min_value=100, max_value=2000, value=500, step=100)

run_full = st.button("Run Full Laser Falcon Analysis", type="primary")

tabs = st.tabs(
    [
        "Data Health",
        "Chain Integrity",
        "IV Skew",
        "IV Surface",
        "OU IV Mean Reversion",
        "Stochastic Volatility",
        "Pressure Mapping",
        "Anomaly & Regime",
        "Temporal Differential",
        "Benchmark / Arbitrage",
        "Exported Reports",
    ]
)

snapshot = None
bench_snapshot = None
load_error = None

try:
    snapshot = load_laser_falcon_snapshot(ticker)
    if benchmark != ticker:
        try:
            bench_snapshot = load_laser_falcon_snapshot(benchmark)
        except FileNotFoundError:
            bench_snapshot = None
except FileNotFoundError as exc:
    load_error = str(exc)

if snapshot is not None:
    expirations = ["Auto"] + list_expirations(snapshot.option_df)
    chosen_exp = None if expiration == "Auto" else expiration
else:
    chosen_exp = None

with tabs[0]:
    st.subheader("Data Health")
    if load_error:
        st.error(load_error)
    elif snapshot is not None:
        st.json(snapshot.data_health)
        st.dataframe(snapshot.option_df.head(20), use_container_width=True)

with tabs[1]:
    st.subheader("Chain Integrity")
    if snapshot is not None:
        integrity = assess_chain_integrity(snapshot.option_df, ticker=ticker, spot_price=snapshot.spot)
        st.metric("Chain health score", integrity.get("chain_health_score"))
        st.write(f"Status: **{integrity.get('status')}**")
        st.json(integrity)
        if integrity.get("strikes_by_expiration"):
            st.table(
                pd.DataFrame(
                    [{"expiration": k, "strikes": v} for k, v in integrity["strikes_by_expiration"].items()]
                )
            )

with tabs[2]:
    st.subheader("IV Skew")
    if snapshot is not None and not snapshot.option_df.empty:
        metrics = compute_skew_metrics(snapshot.option_df, spot=snapshot.spot, expiration=chosen_exp)
        st.json(metrics)
        st.write(
            f"Calls overpriced: **{metrics.get('calls_overpriced_flag')}** | "
            f"Puts overpriced: **{metrics.get('puts_overpriced_flag')}** | "
            f"Skew inverted: **{metrics.get('skew_inversion_flag')}**"
        )
        skew_path = plot_iv_skew(
            snapshot.option_df,
            ticker=ticker,
            spot=snapshot.spot,
            expiration=chosen_exp,
            output_path=Path("outputs/laser_falcon") / f"{ticker}_streamlit_skew.png",
        )
        st.image(str(skew_path))
    else:
        st.warning("No option data loaded.")

with tabs[3]:
    st.subheader("IV Surface")
    if snapshot is not None:
        path, surface = plot_iv_surface(
            snapshot.option_df,
            ticker=ticker,
            spot=snapshot.spot,
            output_path=Path("outputs/laser_falcon") / f"{ticker}_streamlit_surface.png",
        )
        st.json({k: v for k, v in surface.items() if k != "grid"})
        st.image(str(path))

with tabs[4]:
    st.subheader("OU IV Mean Reversion")
    if snapshot is not None:
        skew = compute_skew_metrics(snapshot.option_df, spot=snapshot.spot, expiration=chosen_exp)
        atm = skew.get("atm_iv") or 25.0
        iv0 = float(atm) / 100.0 if atm > 1 else float(atm)
        ou = run_ou_iv_projection(
            iv0=iv0,
            long_run_mean_iv=ou_long_run_iv,
            theta=ou_theta,
            vol_of_vol=vol_of_vol,
            projection_days=projection_days,
            n_paths=n_paths,
        )
        st.json({k: v for k, v in ou.items() if k != "paths_sample"})
        from ou_iv_engine import plot_ou_iv_projection, simulate_ou_iv_paths

        full = simulate_ou_iv_paths(
            iv0=iv0,
            long_run_mean_iv=ou_long_run_iv,
            theta=ou_theta,
            vol_of_vol=vol_of_vol,
            projection_days=projection_days,
            n_paths=n_paths,
        )
        ou_path = plot_ou_iv_projection(
            full,
            ticker=ticker,
            output_path=Path("outputs/laser_falcon") / f"{ticker}_streamlit_ou.png",
        )
        st.image(str(ou_path))

with tabs[5]:
    st.subheader("Stochastic Volatility")
    if snapshot is not None:
        skew = compute_skew_metrics(snapshot.option_df, spot=snapshot.spot, expiration=chosen_exp)
        atm = skew.get("atm_iv") or 25.0
        iv0 = float(atm) / 100.0 if atm > 1 else float(atm)
        stoch = run_stochastic_vol_projection(
            spot0=snapshot.spot,
            atm_iv=iv0,
            projection_days=projection_days,
            n_paths=n_paths,
            rho=stoch_rho,
        )
        st.json(stoch)
        from stochastic_vol_engine import plot_stochastic_vol_projection, simulate_stochastic_vol_paths

        full = simulate_stochastic_vol_paths(
            spot0=snapshot.spot,
            variance0=iv0 ** 2,
            theta=iv0 ** 2,
            rho=stoch_rho,
            projection_days=projection_days,
            n_paths=n_paths,
        )
        stoch_path = plot_stochastic_vol_projection(
            full,
            ticker=ticker,
            output_path=Path("outputs/laser_falcon") / f"{ticker}_streamlit_stoch.png",
        )
        st.image(str(stoch_path))

with tabs[6]:
    st.subheader("Options Pressure Mapping")
    if snapshot is not None:
        skew = compute_skew_metrics(snapshot.option_df, spot=snapshot.spot, expiration=chosen_exp)
        integrity = assess_chain_integrity(snapshot.option_df, ticker=ticker, spot_price=snapshot.spot)
        pm = compute_options_pressure_metrics(
            option_df=snapshot.option_df,
            stock_df=snapshot.stock_df,
            spot=snapshot.spot,
            skew_metrics=skew,
            chain_integrity=integrity,
        )
        st.json(pm)

with tabs[7]:
    st.subheader("Anomaly & Regime")
    if snapshot is not None:
        skew = compute_skew_metrics(snapshot.option_df, spot=snapshot.spot, expiration=chosen_exp)
        pm = compute_options_pressure_metrics(
            option_df=snapshot.option_df,
            stock_df=snapshot.stock_df,
            spot=snapshot.spot,
            skew_metrics=skew,
        )
        bench_skew = None
        if bench_snapshot is not None:
            bench_skew = compute_skew_metrics(bench_snapshot.option_df, spot=bench_snapshot.spot)
        integrity = assess_chain_integrity(snapshot.option_df, ticker=ticker, spot_price=snapshot.spot)
        temporal = compare_ticker_chain_dates(ticker)
        anomaly = detect_anomalies(
            skew_metrics=skew,
            pressure_metrics=pm,
            benchmark_skew=bench_skew,
            data_health=snapshot.data_health,
            chain_integrity=integrity,
            compatibility=temporal.get("compatibility"),
        )
        regime = classify_vol_regime(skew_metrics=skew, pressure_metrics=pm, anomaly=anomaly)
        st.json({"anomaly": anomaly, "regime": regime})

with tabs[8]:
    st.subheader("Temporal Chain Differential")
    if snapshot is not None:
        dates = list_option_chain_dates(ticker)
        st.caption(f"Available chain dates: {', '.join(dates) if dates else 'none'}")
        temporal = compare_ticker_chain_dates(ticker)
        if temporal.get("status") in ("INVALID", "DEGRADED"):
            st.warning(temporal.get("reason", "Temporal comparison confidence reduced"))
        st.json(temporal)

with tabs[9]:
    st.subheader("Benchmark / Vol Arbitrage")
    if snapshot is not None and bench_snapshot is not None:
        report = build_skew_report(snapshot, expiration=chosen_exp, benchmark_snapshot=bench_snapshot)
        st.json(report.get("benchmark", {}))
        arb = detect_vol_arbitrage(
            snapshot.option_df,
            bench_snapshot.option_df,
            target_spot=snapshot.spot,
            benchmark_spot=bench_snapshot.spot,
            target_ticker=ticker,
            benchmark_ticker=benchmark,
        )
        st.json(arb)
    else:
        st.info("Load both target and benchmark tickers for arbitrage detection.")

with tabs[10]:
    st.subheader("Exported Reports")
    if run_full and snapshot is not None:
        with st.spinner("Running Laser Falcon pipeline..."):
            result = run_laser_falcon_analysis(
                ticker,
                benchmark=benchmark,
                projection_days=projection_days,
                ou_theta=ou_theta,
                ou_long_run_iv=ou_long_run_iv,
                vol_of_vol=vol_of_vol,
                stoch_rho=stoch_rho,
                n_paths=n_paths,
            )
        st.success("Analysis complete.")
        st.json({k: v for k, v in result.items() if k not in ("skew", "ou_iv", "stochastic_vol")})
        summary_path = Path(result["summary_md"])
        if summary_path.exists():
            st.markdown(summary_path.read_text(encoding="utf-8"))
    else:
        out_dir = Path("outputs/laser_falcon")
        st.info("Click **Run Full Laser Falcon Analysis** to export MD/JSON/PNG artifacts.")
        if out_dir.exists():
            files = sorted(out_dir.glob(f"{ticker}_*"))
            for f in files:
                st.write(f"`{f}`")
