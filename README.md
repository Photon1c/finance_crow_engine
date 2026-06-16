# Finance Crow Engine

# Recursive Trade Aggregator

A replay and evaluation toolkit for options trade failures. Completed trades are treated as **trajectories** (not just P/L), stored as **lesson packets** in a ledger, and fed into recursive model weights over time. A parallel layer—the **Trading Logistics Driver**—scores trade paths using an autonomous-driving-style reward decomposition.

Two additional market-state engines extend the stack:

- **CanopyEnto** — boundary containment stress and weekly **state-maturity** stance filtering (direction vs trade permission)
- **Capillary Engine** — microscopic noise absorption overlay during cruise mode

**Not a live trading system.** No broker API, no order execution. CSV replay and analysis only.

---

## Project campus

Use this tree as the map. Entry points are marked with ★.

```
recursive_trade_aggregator/
│
├── README.md                              ← navigation guide (this file)
├── recursive_trade_aggregator.code-workspace
│
├── ★ recursive_trade_failure.py           ← main CLI: score failures, update weights, write reports
├── ★ demo_spy_trade.py                    ← end-to-end SPY put logistics-driver demo
├── ★ canopyento_boundary_engine.py        ← boundary stress + weekly stance filter
├── ★ capillary_engine.py                  ← micro-noise absorption overlay (reads CanopyEnto CSV)
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
├── agentearth/                            ← separate 3D agent / chat server (not trade pipeline)
│   ├── serve.py                           ← local dev server + NVIDIA chat proxy
│   ├── admin_genie.py
│   ├── index.html
│   ├── agent_earth.js
│   ├── requirements.txt
│   ├── .env.example
│   └── underthehood/
│       ├── nvidia_api.py
│       ├── nvidia_client.js
│       ├── earth_controls.js
│       ├── space_controls.js
│       ├── camera_intro.js
│       └── iss.js
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
│   └── capillary_engine_latest_{TICKER}.json ← latest capillary snapshot
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
| Run full market-state stack | CanopyEnto → Capillary | see [Quick start](#quick-start) |
| Animate the failure pipeline | `misc_packet_visualizer.py` (via flag) | `python recursive_trade_failure.py --csv --visual` |
| Run the ring-toss pygame demo | `disposition_ring_toss.py` | `python disposition_ring_toss.py` |
| Serve Agent Earth locally | `agentearth/serve.py` | `python agentearth/serve.py` |

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
                      └── outputs/canopyento_boundary_{TICKER}.csv + .md + .json
                                ↓
                      capillary_engine.py
                      ├── brownian_noise, wave_persistence, compression
                      ├── surface_tension, capillary_score, cruise_integrity
                      └── outputs/capillary_engine_{TICKER}.csv + .md + .json
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
| **`agentearth/`** | Separate 3D agent/chat web app with NVIDIA proxy. Not part of the trade failure or logistics driver workflows. |

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

## Dependencies

- Python 3.11+
- `numpy`, `pandas`, `matplotlib` (visualizer)
- Optional: `ffmpeg` on PATH for reliable MP4/GIF export
- Optional: `pygame` for `disposition_ring_toss.py`
- Optional: `agentearth/requirements.txt` for the Agent Earth server
