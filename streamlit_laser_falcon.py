"""Streamlit UI for Laser Falcon options research (CSV replay only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from laser_falcon_data_adapter import list_expirations, load_laser_falcon_snapshot
from laser_falcon_primary_engine import build_skew_report, run_laser_falcon_analysis, run_ou_iv_projection, run_stochastic_vol_projection
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

projection_days = st.slider("Projection days", min_value=5, max_value=90, value=30)
ou_theta = st.slider("OU theta (mean reversion speed)", min_value=0.5, max_value=12.0, value=4.0, step=0.5)
ou_long_run_iv = st.slider("OU long-run IV", min_value=0.05, max_value=1.0, value=0.25, step=0.01)
vol_of_vol = st.slider("Vol-of-vol (OU sigma)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
stoch_rho = st.slider("Stochastic vol rho (price-vol correlation)", min_value=-0.99, max_value=0.99, value=-0.70, step=0.01)
n_paths = st.slider("Simulation paths", min_value=100, max_value=2000, value=500, step=100)

run_full = st.button("Run Full Laser Falcon Analysis", type="primary")

tabs = st.tabs(
    [
        "Data Health",
        "IV Skew",
        "IV Surface",
        "OU IV Mean Reversion",
        "Stochastic Volatility",
        "Benchmark Comparison",
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
    if expiration == "Auto" and len(expirations) > 1:
        chosen_exp = None
    else:
        chosen_exp = expiration if expiration != "Auto" else None
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
    st.subheader("IV Skew")
    if snapshot is not None and not snapshot.option_df.empty:
        metrics = compute_skew_metrics(snapshot.option_df, spot=snapshot.spot, expiration=chosen_exp)
        st.json(metrics)
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

with tabs[2]:
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

with tabs[3]:
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

with tabs[4]:
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

with tabs[5]:
    st.subheader("Benchmark Comparison")
    if snapshot is not None:
        report = build_skew_report(snapshot, expiration=chosen_exp, benchmark_snapshot=bench_snapshot)
        st.json(report.get("benchmark", {"note": "Benchmark unavailable"}))

with tabs[6]:
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
