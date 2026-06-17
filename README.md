# Finance Crow Engine

A replay and evaluation toolkit for options trade failures and market-state observation. Completed trades are treated as **trajectories** (not just P/L), stored as **lesson packets** in a ledger, and fed into recursive model weights over time. A parallel layer—the **Trading Logistics Driver**—scores trade paths using an autonomous-driving-style reward decomposition.

Market-state engines extend the stack:

- **CanopyEnto** — pressure-field observer: boundary containment stress, observer differential metrics, weekly stance filtering
- **Capillary Engine** — microscopic noise absorption overlay during cruise mode
- **Pressure Field Dashboard** — multi-sensor HTML dashboard with LRP and rate-of-change derivatives
- **Laser Falcon** — CSV options research: IV skew/surface, OU mean reversion, stochastic vol cones
- **TRPR** — sacred packet ontology (`TRPR/ontology/packet_ontology.yaml`) shared across domains

**Not a live trading system.** No broker API, no order execution. CSV replay and analysis only.

> **LRP doctrine:** Baseline LRP = pressure signal; LRP_adjusted (experimental) = pressure after restoration/capillary/hysteresis/observer modifiers.

> **Elastic rebound layer:** Maps gamma locking, hidden reservoir pressure, and false-stability risk into finance-local metrics. Sacred ontology untouched.

> **Laser Falcon:** CSV-driven options research engine — IV skew/surface, OU mean reversion, stochastic vol projections. Sacred ontology untouched; map inward via `config/pressure_ontology.yaml`. Temporal analysis requires comparable contract universes; Contract Universe Drift is detected and reported. Sparse IPO chains may support skew but not surface or temporal compare.

> **For LLM / agent onboarding:** read [`summary.md`](summary.md) first — repo map, architecture, sacred ontology rules, and navigation guide.

---

## Project campus

Use this tree as the map. Entry points are marked with ★.

```
finance_crow_engine/
│
├── README.md                              ← human navigation guide (this file)
├── summary.md                             ← LLM onboarding: architecture, file tree, rules
├── archive/
│   ├── log.md                             ← conversation milestones
│   ├── transition_framework.md            ← universal transition framework archive
│   └── transition_framework.tex             ← formal equations (LaTeX)
├── recursive_trade_aggregator.code-workspace
│
├── ★ recursive_trade_failure.py           ← main CLI: score failures, update weights, write reports
├── ★ demo_spy_trade.py                    ← end-to-end SPY put logistics-driver demo
├── ★ canopyento_boundary_engine.py        ← boundary stress + weekly stance filter
├── ★ capillary_engine.py                  ← micro-noise absorption overlay (reads CanopyEnto CSV)
├── ★ pressure_field_dashboard.py          ← HTML pressure-field dashboard (MACD/RSI/CVD/VWAP/gamma)
├── pressure_field_schema.py               ← stable latest JSON snapshot keys
├── pressure_field_derivatives.py          ← LRP, rate-of-change derivatives, alerts
├── pressure_field_physics.py              ← orchestrates restoration/capillary/attractor/hysteresis/entropy
├── elastic_rebound_engine.py              ← gamma strain, hidden reservoir, false stability
├── ★ laser_falcon_primary_engine.py       ← IV skew/surface, OU + stoch vol orchestration
├── laser_falcon_data_adapter.py           ← normalized stock/option chain adapter
├── volatility_skew_engine.py              ← skew metrics + benchmark compare
├── volatility_surface_engine.py           ← 3D IV surface (sparse-tolerant)
├── implied_vol_solver.py                  ← Black-Scholes IV inversion
├── ou_iv_engine.py                        ← OU IV mean reversion paths
├── stochastic_vol_engine.py               ← Heston-like vol cones
├── laser_falcon_regime_mapper.py          ← map vol metrics to pressure vocabulary
├── temporal_chain_differential_engine.py  ← yesterday vs today chain pressure deltas
├── chain_integrity_engine.py              ← single-snapshot chain health diagnostics
├── chain_compatibility_engine.py          ← temporal CUD / overlap guardrails
├── ★ streamlit_laser_falcon.py            ← interactive Laser Falcon UI
├── restoration_field_engine.py            ← F_r restoring field, D_c dissipation capacity
├── capillary_wave_engine.py               ← A_f oscillation amplitude, C_w capillary wave score
├── attractor_engine.py                    ← equilibrium field strength, deviation
├── hysteresis_engine.py                   ← stress memory, recursive pressure carryover
├── entropy_engine.py                      ← long-term degradation (entropy_score)
├── observer_feedback_engine.py            ← observer coupling / effective_pressure
├── field_regime_engine.py                 ← sacred named field regimes
├── config/pressure_ontology.yaml          ← market domain mapping (references sacred ontology)
├── TRPR/                                  ← Temporal Relational Packet Reconstructor root
│   └── ontology/
│       ├── packet_ontology.yaml           ← SACRED shared packet vocabulary
│       ├── ONTOLOGY_CHARTER.md            ← drift prevention charter
│       └── packet_ontology_loader.py      ← optional read-only loader
│
├── tests/                                 ← unit tests (pressure field + sacred ontology + physics)
│   ├── test_pressure_field.py
│   ├── test_pressure_field_physics.py
│   ├── test_lrp_loop_closure.py
│   ├── test_elastic_rebound.py
│   ├── test_laser_falcon.py
│   ├── test_chain_integrity.py
│   ├── test_chain_compatibility.py
│   └── test_packet_ontology.py
│
├── data_loader.py                         ← shared stock + option CSV loading
├── recursive_weight_engine.py             ← JSON weight load/save, packet scoring, online updates
├── packet_persistence.py                  ← ledger append + report file paths
├── packet_similarity.py                   ← investigator-style similarity search over ledger
│
├── trade_trajectory.py                    ← TradeTrajectory dataclass (entry/exit path)
├── trajectory_reward.py                   ← RewardSchema + trajectory classifications
├── regime_classifier.py                   ← road-condition regime labels from price history
├── logistics_driver.py                    ← orchestrates regime + report + lesson text
├── agentlightning_adapter.py              ← RL transition stub (no AgentLightning required)
│
├── misc_packet_visualizer.py              ← six-phase pipeline animation (MP4/GIF)
├── disposition_ring_toss.py               ← standalone pygame demo (unrelated metaphor)
│
├── outputs/                               ← generated artifacts (safe to regenerate)
│   ├── recursive_packets_{TICKER}.csv     ← packet ledger
│   ├── recursive_weights_{TICKER}.json    ← component weights per ticker
│   ├── latest_report_{TICKER}.md          ← failure aggregator report
│   ├── canopyento_boundary_{TICKER}.csv   ← boundary + weekly stance time series
│   ├── canopyento_boundary_{TICKER}.md    ← boundary rupture report
│   ├── canopyento_weekly_stance_{TICKER}.json ← latest stance packet (dashboard-ready)
│   ├── capillary_engine_{TICKER}.csv      ← capillary metrics time series
│   ├── capillary_engine_{TICKER}.md       ← capillary report
│   ├── capillary_engine_latest_{TICKER}.json ← latest capillary snapshot
│   ├── pressure_field_dashboard_{TICKER}.html ← interactive pressure-field dashboard
│   ├── pressure_field_{TICKER}.csv        ← pressure field + LRP + derivative time series
│   ├── pressure_field_{TICKER}.md         ← markdown report with LRP and ROC alerts
│   └── pressure_field_latest_{TICKER}.json  ← stable latest snapshot (dashboard)
│
├── recursive_weights_SPY.json             ← legacy/root copies (prefer outputs/)
├── recursive_weights_UEC.json
├── recursive_trade_failure_SPY.mp4        ← last visual export (from --visual)
└── recursive_trade_failure_SPY.gif
```

External data (not in repo; configure via CLI flags):

```
F:/inputs/
├── stocks/{TICKER}.csv                    ← OHLCV history (Date, Close/Last, Volume, Open, High, Low)
└── options/log/{ticker}/{date}/
    └── {ticker}_quotedata.csv             ← option chain snapshot
```

---

## Navigate by purpose

### I want to run something

| Goal | Start here | Command |
|------|------------|---------|
| Score trade failures and update weights | `recursive_trade_failure.py` | `python recursive_trade_failure.py --csv` |
| Demo the logistics driver on SPY | `demo_spy_trade.py` | `python demo_spy_trade.py` |
| Detect boundary containment stress | `canopyento_boundary_engine.py` | `python canopyento_boundary_engine.py` |
| Measure micro-noise absorption (Capillary) | `capillary_engine.py` | `python capillary_engine.py --ticker SPY` |
| View pressure-field HTML dashboard | `pressure_field_dashboard.py` | `python pressure_field_dashboard.py --ticker SPY --open` |
| Options IV research (Laser Falcon) | `laser_falcon_primary_engine.py` | `python laser_falcon_primary_engine.py --ticker SPCX --benchmark SPY` |
| Interactive Laser Falcon UI | `streamlit_laser_falcon.py` | `streamlit run streamlit_laser_falcon.py` |
| Run full market-state stack | CanopyEnto → Capillary | see [Quick start](#quick-start) |
| Animate the failure pipeline | `misc_packet_visualizer.py` (via flag) | `python recursive_trade_failure.py --csv --visual` |
| Run the ring-toss pygame demo | `disposition_ring_toss.py` | `python disposition_ring_toss.py` |

### I want to understand a pipeline

**Recursive failure pipeline**

```
data_loader.py → recursive_trade_failure.py
                      ├── recursive_weight_engine.py   (score + update weights)
                      ├── packet_persistence.py        (ledger + report paths)
                      ├── packet_similarity.py         (similar historical packets)
                      └── misc_packet_visualizer.py    (--visual only)
```

**Trading Logistics Driver**

```
data_loader.py → demo_spy_trade.py
                      ├── trade_trajectory.py          (completed trade path)
                      ├── trajectory_reward.py         (reward decomposition)
                      ├── regime_classifier.py         (road-condition label)
                      ├── logistics_driver.py          (report + lesson)
                      └── agentlightning_adapter.py      (RL stub export)
```

**CanopyEnto + Capillary stack**

```
data_loader.py → canopyento_boundary_engine.py
                      ├── B_s, E_i, rupture_pressure_score, regime_label
                      ├── regime_persistence, rupture_probability
                      ├── weekly stance vector (direction vs trade permission)
                      ├── observer differential (T_a, R_o, T_v, observer_profile)
                      └── outputs/canopyento_boundary_{TICKER}.csv + .md + .json
                                ↓
                      capillary_engine.py
                      ├── brownian_noise, wave_persistence, compression
                      ├── surface_tension, capillary_score, cruise_integrity
                      └── outputs/capillary_engine_{TICKER}.csv + .md + .json

pressure_field_dashboard.py (parallel read — uses stock + option CSV directly)
                      ├── MACD, RSI, CVD proxy, Volume, VWAP, gamma flip
                      ├── LRP (Latent Rupture Potential) + rate-of-change derivatives
                      ├── physics layer: F_r, D_c, A_f, C_w, field_regime (via pressure_field_physics)
                      ├── merges CanopyEnto boundary + observer differential metrics
                      └── outputs/pressure_field_*.{html,csv,md,json}
```

**CanopyEnto** detects stored pressure and boundary regime. **Capillary** detects whether microscopic noise is still being absorbed. A market can remain in cruise mode while its Brownian layer becomes increasingly unstable.

---

## Quick start

```powershell
# Recursive failure aggregator (CSV market data)
python recursive_trade_failure.py --csv

# Demo trades only (no CSV)
python recursive_trade_failure.py --no-csv

# Pipeline animation (MP4 + GIF)
python recursive_trade_failure.py --csv --visual

# Trading Logistics Driver — SPY put demo
python demo_spy_trade.py

# CanopyEnto boundary rupture engine (defaults to SPY)
python canopyento_boundary_engine.py

# Capillary Engine — reads CanopyEnto output (run CanopyEnto first)
python capillary_engine.py --ticker SPY

# Pressure Field Dashboard — HTML + stable JSON snapshot
python pressure_field_dashboard.py --ticker SPY --open

# Full market-state stack
python canopyento_boundary_engine.py --ticker SPY
python capillary_engine.py --ticker SPY
```

---

## Module reference

### Entry points

| File | Role |
|------|------|
| **`recursive_trade_failure.py`** | Main CLI. Scores trade failure packets, updates recursive weights, appends the ledger, finds similar historical packets, and writes a Markdown report. Supports investigator queries (`--similar-to`, `--filter`) and optional visualization (`--visual`). |
| **`demo_spy_trade.py`** | End-to-end demo of the Trading Logistics Driver on SPY. Loads CSV snapshot, picks a cheap put (ask ≤ $1.50 by default), anchors entry to today, plans exit 7–14 days out, and prints a Markdown evaluation report. |
| **`canopyento_boundary_engine.py`** | Boundary containment stress engine. Computes `B_s`, `E_i`, rupture pressure, regime labels, and a weekly stance filter that separates **directional bias** from **trade permission**. Forecasts state maturity, not price. |
| **`capillary_engine.py`** | Capillary instability overlay. Reads CanopyEnto CSV output and measures whether microscopic noise is being absorbed or beginning to persist and amplify toward rupture. |
| **`pressure_field_dashboard.py`** | Multi-sensor HTML dashboard. Loads stock + option CSV via `data_loader`, computes MACD/RSI/CVD/VWAP/gamma flip, merges CanopyEnto observer metrics, writes `outputs/pressure_field_dashboard_{TICKER}.html` and stable JSON snapshot. |

### Pressure Field Dashboard

**Run**

```powershell
python pressure_field_dashboard.py --ticker SPY --open
```

**Required inputs** (defaults shown; override with `--stock-dir` / `--option-dir`):

| Input | Default path |
|-------|----------------|
| Stock OHLCV CSV | `F:/inputs/stocks/{TICKER}.csv` |
| Option chain snapshot | `F:/inputs/options/log/{ticker}/{date}/{ticker}_quotedata.csv` |

If the option chain is missing, the dashboard still runs; gamma flip fields are written as `null` with `gamma_regime: NO_CHAIN`.

**Outputs**

| File | Contents |
|------|----------|
| `outputs/pressure_field_dashboard_{TICKER}.html` | Self-contained interactive dashboard (LRP card + ROC alerts) |
| `outputs/pressure_field_{TICKER}.csv` | Time series with LRP, derivatives, and sensor regimes |
| `outputs/pressure_field_{TICKER}.md` | Markdown report with LRP snapshot and alerts |
| `outputs/pressure_field_latest_{TICKER}.json` | Stable flat snapshot for downstream ingestion |

Both `pressure_field_latest_{TICKER}.json` and `canopyento_weekly_stance_{TICKER}.json` share the same core keys:

`ticker`, `timestamp`, `close`, `macd_regime`, `rsi_regime`, `cvd_regime`, `volume_ratio`, `vwap_distance_pct`, `gamma_flip`, `gamma_flip_distance_pct`, `canopy_regime`, `T_a`, `T_a_norm`, `T_a_regime`, `R_o`, `T_v`, `observer_profile`, `LRP`, `LRP_regime`, `d_canopy_pressure`, `dd_canopy_pressure`, `d_R_o`, `d_T_v`, `d_gamma_flip_distance`, `d_vwap_distance`

**Observer differential variables (CanopyEnto)**

| Variable | Meaning |
|----------|---------|
| **T_a** | Transitional acceleration — second derivative of rupture pressure (`d²P/dt²`). Positive → takeoff beginning; near zero → cruise; negative → thrust loss; strongly negative → dissipation cascade. |
| **R_o** | Observational resolution — ability to perceive latent instability before visible rupture (0–1). Higher → earlier detection. |
| **T_v** | Visibility horizon — sessions before rupture becomes visible to this observer; `T_v = f(R_o)`, clamped 0–10 sessions. |
| **observer_profile** | Tier label derived from R_o: `passenger` (low), `pilot` (medium), `mechanic` (high). |

**Latent Rupture Potential (LRP)**

| Variable | Meaning |
|----------|---------|
| **LRP** | Latent Rupture Potential — weighted pre-rupture score in [0, 1] (**canonical pressure signal**). Raw components are summed with mild absorption dampening, then mapped through a sigmoid. |
| **LRP_regime** | `STABLE` (<0.30) · `PRESSURE_BUILDING` · `PRE_RUPTURE` · `RUPTURE_IMMINENT` (≥0.85) — ranges from sacred `TRPR/ontology/packet_ontology.yaml` |
| **LRP_adjusted** | **Experimental** — restoration-adjusted sibling metric. Same sigmoid calibration applied after restoration/capillary/hysteresis/observer modifiers. **Do not treat as canonical.** |

**Rate-of-change derivatives** (first row defaults to 0.0): `d_macd_pressure`, `d_rsi_energy`, `d_cvd_force`, `d_volume_energy`, `d_vwap_attractor_distance`, `d_gamma_flip_distance`, `d_canopy_pressure`, `d_observability_R_o`, `d_visibility_horizon_T_v`, plus `dd_canopy_pressure`, `dd_observability_R_o`, `dd_coherence_proxy`.

**CVD note:** CVD in this stack is a **signed-volume proxy** (close direction × volume), not tick-true cumulative volume delta.

**Ontology layers**

| Layer | Path | Role |
|-------|------|------|
| Sacred ontology | `TRPR/ontology/packet_ontology.yaml` | Shared packet primitives — do not drift |
| Domain mapping | `config/pressure_ontology.yaml` | Market sensors → sacred primitives (references only) |
| Charter | `TRPR/ontology/ONTOLOGY_CHARTER.md` | Drift-prevention rules |

### Recursive failure pipeline

| File | Role |
|------|------|
| **`data_loader.py`** | Shared market data layer. Loads stock CSVs and option chains, resolves latest prices and chain dates, selects near-ATM or max-premium contracts, validates strike/spot alignment, and builds synchronized market snapshots. |
| **`recursive_weight_engine.py`** | Loads/saves JSON weights per ticker. Scores packets with weighted component penalties, applies online weight updates after each run, and formats theory lines for the report. |
| **`packet_persistence.py`** | Appends scored packets to `outputs/recursive_packets_{TICKER}.csv` and saves Markdown reports. |
| **`packet_similarity.py`** | Investigator-style similarity search over the ledger: weighted Euclidean distance, optional `--similar-to` component boost, and ledger filters (`failure_type`, `status`, etc.). |
| **`misc_packet_visualizer.py`** | Six-phase pipeline animation (reality → sanitize → gate → inference → case → outcome). Called by `recursive_trade_failure.py --visual`; exports MP4 and GIF. |

### Trading Logistics Driver

| File | Role |
|------|------|
| **`trade_trajectory.py`** | `TradeTrajectory` dataclass for a completed option trade: entry/exit times, premium, underlying path, greeks, and derived fields (`pnl_pct`, `moneyness`, `was_direction_correct()`, etc.). |
| **`trajectory_reward.py`** | `RewardSchema` dataclass with component scores (direction, timing, magnitude, volatility, exit), weighted `total_reward()`, and trajectory classifications. |
| **`regime_classifier.py`** | Road-condition labels from underlying return, realized vol, IV change, and optional flow/volume: `cruise_descent`, `rupture`, `chop`, `compression`, `reversal`, `unknown`. |
| **`logistics_driver.py`** | Orchestration: given a trajectory + reward schema, classifies regime, builds a structured `LogisticsReport`, and formats Markdown. |
| **`agentlightning_adapter.py`** | Stub adapter for future RL integration. Exports `{state, action, reward, next_state, done, metadata}` transitions without requiring AgentLightning installed. |

### Other

| File | Role |
|------|------|
| **`disposition_ring_toss.py`** | Standalone pygame visualization (criminal-justice case-flow metaphor). Unrelated to the trade aggregator pipeline. |

---

## CLI reference

### `recursive_trade_failure.py`

```powershell
python recursive_trade_failure.py --csv
python recursive_trade_failure.py --no-csv
python recursive_trade_failure.py --csv --visual --ticker SPY
python recursive_trade_failure.py --no-csv --similar-to timing --filter failure_type=Timing
```

| Flag | Description |
|------|-------------|
| `--csv [DIR]` | Load stock/option CSVs (default dirs or `DIR/stocks` + `DIR/options/log`) |
| `--no-csv` | Built-in demo trades (SPY Put, UEC Weekly Call) |
| `--ticker SPY` | Primary ticker for CSV and visual labels |
| `--visual` | Render packet pipeline animation |
| `--similar-to` | Focus similarity on one component (`direction`, `timing`, `magnitude`, …) |
| `--filter KEY=VALUE` | Ledger filter (repeatable) |

### `demo_spy_trade.py`

```powershell
python demo_spy_trade.py
python demo_spy_trade.py --max-premium 1.50 --min-exit-days 7 --max-exit-days 14
python demo_spy_trade.py --stock-dir "F:/inputs/stocks" --option-dir "F:/inputs/options/log"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-premium` | `1.50` | Maximum option **ask** at entry |
| `--min-exit-days` | `7` | Minimum hold before exit |
| `--max-exit-days` | `14` | Maximum hold window; picks best chain bid in range |
| `--stock-dir` | `F:/inputs/stocks` | Stock CSV directory |
| `--option-dir` | `F:/inputs/options/log` | Option chain root |

### `canopyento_boundary_engine.py`

```powershell
python canopyento_boundary_engine.py
python canopyento_boundary_engine.py --ticker SPY --lookback 20 --tolerance 0.003
python canopyento_boundary_engine.py --ticker SPY --weekly-window 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `SPY` | Stock ticker symbol |
| `--lookback` | `20` | Rolling window for boundary detection |
| `--tolerance` | `0.003` | Boundary proximity tolerance (fraction) |
| `--volume-window` | `20` | Rolling window for average volume |
| `--weekly-window` | `5` | Trailing sessions for weekly stance evaluation |
| `--output` | `outputs/canopyento_boundary_{TICKER}.csv` | Metrics CSV path |
| `--report` | `outputs/canopyento_boundary_{TICKER}.md` | Markdown report path |
| `--stance-json` | `outputs/canopyento_weekly_stance_{TICKER}.json` | Latest stance JSON path |
| `--stock-dir` | `F:/inputs/stocks` | Stock CSV directory |

**Boundary metrics:** `B_s` (boundary stress frequency), `E_i` (energy injection rate), `rupture_pressure_score = B_s × E_i`.

**Boundary regimes:** `IDLE`, `PRESSURE_BUILDING`, `ENERGY_INJECTION`, `CONTAINMENT_STRESS`, `RUPTURE_CANDIDATE`.

**Weekly stance vector:** direction, timing, magnitude, volatility, packet completion, absorption, hidden uncertainty, continuation, `regime_persistence`, `rupture_probability`.

**Stance quadrants:** `bearish|bullish / actionable|unresolved` — separates directional bias from trade permission.

**Gate stances:** `WAIT / PACKET BUFFERING`, `BEARISH BUT ABSORBED`, `ACTIONABLE DIRECTIONAL STANCE`, `LOW-CONFIDENCE CRUISE MODE`.

### `capillary_engine.py`

```powershell
python capillary_engine.py --ticker SPY
python capillary_engine.py --ticker SPY --input outputs/canopyento_boundary_SPY.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `SPY` | Stock ticker symbol |
| `--input` | `outputs/canopyento_boundary_{TICKER}.csv` | CanopyEnto CSV input |
| `--output` | `outputs/capillary_engine_{TICKER}.csv` | Capillary metrics CSV path |
| `--report` | `outputs/capillary_engine_{TICKER}.md` | Markdown report path |
| `--json` | `outputs/capillary_engine_latest_{TICKER}.json` | Latest snapshot JSON path |

**Core formula:**

```
capillary_score = (brownian_noise × wave_persistence × compression) / max(surface_tension, 0.05)
```

**Capillary regimes:** `ABSORBING_NOISE`, `CRUISE_SURFACE_ACTIVE`, `SURFACE_RIPPLING`, `PINCH_OFF_WATCH`, `CAPILLARY_RUPTURE_RISK`.

**Derived fields:** `cruise_integrity`, `pinch_off_risk`, `combined_read` (synthesizes CanopyEnto + Capillary context).

---

## Architecture

**Recursive failure loop:**

```
Market State (CSV) → Forecast / Packet → Outcome → Failure Attribution → Model Update → Future Forecast
                              ↓
                    Ledger + Weights + Similarity
```

**Logistics Driver** (parallel evaluation layer):

```
TradeTrajectory + RewardSchema → Regime → Report + Lesson + RL stub transition
```

**CanopyEnto + Capillary** (market-state maturity stack):

```
Stock CSV → Rolling boundaries + volume
              → B_s, E_i, rupture_pressure, regime_persistence
              → weekly stance vector (direction vs permission)
              → gate stance + recommended action
                        ↓
              Capillary: brownian_noise × wave_persistence × compression
                         ÷ surface_tension
              → cruise_integrity, pinch_off_risk, combined_read
```

Core ideas:

- A losing trade can have a **correct directional thesis** but a **wrong vehicle**. P/L alone is not the reward—**trajectory correctness** is.
- A forecast should not only predict direction. It should estimate whether the observed system has **finished becoming the thing being predicted**.
- CanopyEnto can show **stored pressure building** while Capillary still shows **microscopic noise being absorbed**—macro stress and micro stability can diverge.

---

## Tests

```powershell
python -m unittest tests.test_pressure_field tests.test_packet_ontology tests.test_pressure_field_physics tests.test_lrp_loop_closure tests.test_elastic_rebound tests.test_laser_falcon tests.test_chain_integrity tests.test_chain_compatibility -v
```

## Dependencies

- Python 3.11+
- `numpy`, `pandas`, `matplotlib` (visualizer), `pyyaml` (TRPR ontology loader)
- Optional: `ffmpeg` on PATH for reliable MP4/GIF export
- Optional: `pygame` for `disposition_ring_toss.py`
