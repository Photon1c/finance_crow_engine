# Finance Crow Engine — LLM Onboarding Summary

**Not a live trading system.** No broker API, no order execution. CSV replay and analysis only.

> **LRP doctrine:** Baseline LRP = pressure signal; LRP_adjusted (experimental) = pressure after restoration/capillary/hysteresis/observer modifiers.

> **Elastic rebound layer:** Maps gamma locking, hidden reservoir pressure, and false-stability risk into finance-local metrics. Sacred ontology untouched.

> **Laser Falcon:** CSV-driven options research engine — IV skew/surface, OU mean reversion, stochastic vol projections. Maps inward via `config/pressure_ontology.yaml` only. Option chain temporal analysis requires comparable contract universes; Contract Universe Drift (CUD) is detected and reported.

Read this file first when working in this repository. It is the fast orientation layer for agents and future LLMs. Human-oriented CLI detail lives in [`README.md`](README.md).

---

## What this repo is

**Finance Crow Engine** is a CSV-replay analysis toolkit — not a live trading system. It treats market behavior and trade outcomes as **packets** moving through relational systems, not as isolated events.

The repo has five cooperating layers:

1. **Recursive trade failure loop** — score failed option trades, store lesson packets, update recursive weights
2. **Trading Logistics Driver** — evaluate completed trade trajectories with autonomous-driving-style reward decomposition
3. **Market-state observation stack** — CanopyEnto (pressure observer) → Capillary (micro-noise absorption) → Pressure Field Dashboard (multi-sensor HTML + LRP)
4. **Laser Falcon** — options IV skew/surface, OU + stochastic vol research (CSV replay; no broker)
5. **TRPR sacred ontology** — shared packet vocabulary that all future engines map *into* (never redefine locally)

---

## Architectural shift (read before editing)

From [`archive/log.md`](archive/log.md):

- event reconstruction → **packet reconstruction**
- indicator engine → **pressure field observer** (CanopyEnto)
- repo monitor → **autonomous maintenance ecology** (OpenClaw — external concept, not in this repo yet)
- introduced **IDI** (Inward Drift Index) and **LRP** (Latent Rupture Potential) in sacred ontology
- separating **Custodian** vs **Security Guard** agent roles (design direction)
- architecture moving toward **universal transition mathematics**

---

## Sacred ontology — do not drift

`TRPR/ontology/packet_ontology.yaml` is **sacred**. Rules in `TRPR/ontology/ONTOLOGY_CHARTER.md`:

| Rule | Meaning |
|------|---------|
| Single source of truth | Primitives, derived metrics, regimes, lifecycle states live only in the sacred YAML |
| Map inward | Domain engines translate local metrics → sacred keys; never fork vocabulary in Python or side YAML |
| Read-only at runtime | Load and reference; never mutate at runtime |
| Generalize before adding | New ontology entries need cross-domain justification |

**Layers:**

| Layer | Path | Role |
|-------|------|------|
| Sacred ontology | `TRPR/ontology/packet_ontology.yaml` | Canonical packet vocabulary (schema v0.1.0) |
| Read-only loader | `TRPR/ontology/packet_ontology_loader.py` | Optional import — engines are **not** required to depend on it yet |
| Domain mapping | `config/pressure_ontology.yaml` | Finance-crow market sensors → sacred primitives |
| Charter | `TRPR/ontology/ONTOLOGY_CHARTER.md` | Drift prevention |

```python
# Optional — do not hardwire every engine to this yet
from TRPR.ontology.packet_ontology_loader import load_packet_ontology, classify_regime
ontology = load_packet_ontology()
regime = classify_regime(0.42)  # uses sacred LRP regime ranges
```

---

## File tree

```
finance_crow_engine/
│
├── README.md                              ← human navigation + CLI reference
├── summary.md                             ← this file (LLM onboarding)
├── LICENSE
├── .gitignore
├── recursive_trade_aggregator.code-workspace
│
├── ★ ENTRY POINTS
│   ├── recursive_trade_failure.py         ← main failure CLI, weights, ledger, --visual
│   ├── demo_spy_trade.py                  ← Trading Logistics Driver SPY put demo
│   ├── canopyento_boundary_engine.py      ← boundary stress + observer differential (T_a, R_o, T_v)
│   ├── capillary_engine.py                ← micro-noise absorption (reads CanopyEnto CSV)
│   └── pressure_field_dashboard.py        ← HTML dashboard: MACD/RSI/CVD/VWAP/gamma + LRP
│
├── MARKET-STATE SUPPORT
│   ├── pressure_field_derivatives.py      ← LRP (sigmoid-calibrated), ROC derivatives, alerts
│   ├── pressure_field_physics.py          ← orchestrates Phase 1–3 physics engines
│   ├── elastic_rebound_engine.py          ← gamma strain, hidden reservoir, false stability
│   ├── ★ laser_falcon_primary_engine.py   ← IV skew/surface, OU + stoch vol orchestration
│   ├── laser_falcon_data_adapter.py       ← normalized stock/option chain adapter
│   ├── volatility_skew_engine.py          ← skew metrics + benchmark compare
│   ├── volatility_surface_engine.py       ← 3D IV surface (sparse-tolerant)
│   ├── implied_vol_solver.py              ← Black-Scholes IV inversion
│   ├── ou_iv_engine.py                    ← OU IV mean reversion paths
│   ├── stochastic_vol_engine.py           ← Heston-like vol cones
│   ├── laser_falcon_regime_mapper.py      ← map vol metrics to pressure vocabulary
│   ├── options_pressure_mapper.py         ← gamma/vol-expansion/skew/dealer pressure scores
│   ├── anomaly_detection_engine.py        ← IPO/sparse chain anomaly labels
│   ├── regime_detection_engine.py         ← LOW/HIGH/PANIC vol regime classification
│   ├── volatility_arbitrage_detector.py   ← cross-ticker IV dislocation detector
│   ├── temporal_chain_differential_engine.py ← yesterday vs today chain pressure deltas
│   ├── chain_integrity_engine.py          ← single-snapshot chain health diagnostics
│   ├── chain_compatibility_engine.py      ← temporal CUD / overlap guardrails
│   ├── projection_range_engine.py         ← 7–180 day projection presets
│   └── ★ streamlit_laser_falcon.py        ← interactive Laser Falcon UI
│   ├── restoration_field_engine.py        ← F_r, D_c, restoration_ratio
│   ├── capillary_wave_engine.py           ← A_f, wave_persistence, C_w
│   ├── attractor_engine.py                ← equilibrium_field_strength, deviation
│   ├── hysteresis_engine.py               ← historical_stress_memory, carryover
│   ├── entropy_engine.py                  ← entropy_score (long-term degradation)
│   ├── observer_feedback_engine.py        ← observer_feedback_score, effective_pressure
│   ├── field_regime_engine.py             ← sacred named field regimes
│   ├── pressure_field_schema.py           ← stable JSON snapshot keys (shared with CanopyEnto)
│   └── data_loader.py                     ← stock + option CSV loading (shared)
│
├── RECURSIVE FAILURE PIPELINE
│   ├── recursive_weight_engine.py         ← JSON weights, packet scoring, online updates
│   ├── packet_persistence.py              ← ledger append + report paths
│   ├── packet_similarity.py               ← investigator similarity search over ledger
│   └── misc_packet_visualizer.py          ← six-phase pipeline animation (MP4/GIF)
│
├── LOGISTICS DRIVER
│   ├── trade_trajectory.py                ← TradeTrajectory dataclass
│   ├── trajectory_reward.py               ← RewardSchema + classifications
│   ├── regime_classifier.py               ← road-condition regime labels
│   ├── logistics_driver.py                ← report + lesson orchestration
│   └── agentlightning_adapter.py          ← RL transition stub (no AgentLightning required)
│
├── TRPR/                                  ← Temporal Relational Packet Reconstructor
│   ├── __init__.py
│   └── ontology/
│       ├── packet_ontology.yaml           ← SACRED — do not drift
│       ├── ONTOLOGY_CHARTER.md
│       ├── packet_ontology_loader.py      ← optional read-only loader
│       └── __init__.py
│
├── config/
│   └── pressure_ontology.yaml             ← market domain → sacred primitive mapping
│
├── tests/
│   ├── test_pressure_field.py             ← LRP, derivatives, gamma edge cases, schema (21 tests)
│   └── test_packet_ontology.py            ← sacred ontology loader (5 tests)
│
├── archive/
│   ├── log.md
│   ├── transition_framework.md            ← universal transition framework + A_d frontier
│   └── transition_framework.tex           ← LaTeX equations
│
├── outputs/                               ← generated artifacts (safe to regenerate)
│   ├── recursive_packets_{TICKER}.csv
│   ├── recursive_weights_{TICKER}.json
│   ├── latest_report_{TICKER}.md
│   ├── canopyento_boundary_{TICKER}.{csv,md}
│   ├── canopyento_weekly_stance_{TICKER}.json
│   ├── capillary_engine_{TICKER}.{csv,md}
│   ├── capillary_engine_latest_{TICKER}.json
│   ├── pressure_field_dashboard_{TICKER}.html
│   ├── pressure_field_{TICKER}.{csv,md}
│   ├── pressure_field_latest_{TICKER}.json
│   └── lrp_debug_{TICKER}.json
│
├── OTHER
│   ├── disposition_ring_toss.py             ← standalone pygame demo (unrelated)
│   ├── recursive_weights_SPY.json         ← legacy root copies (prefer outputs/)
│   └── recursive_weights_UEC.json
│
└── EXTERNAL DATA (not in repo)
    F:/inputs/stocks/{TICKER}.csv
    F:/inputs/options/log/{ticker}/{date}/{ticker}_quotedata.csv
```

---

## Pipeline map

### 1. Recursive failure loop

```
data_loader.py → recursive_trade_failure.py
    ├── recursive_weight_engine.py   (score + update weights)
    ├── packet_persistence.py        (ledger + report paths)
    ├── packet_similarity.py         (similar historical packets)
    └── misc_packet_visualizer.py    (--visual only)
```

**Run:** `python recursive_trade_failure.py --csv`

### 2. Trading Logistics Driver

```
data_loader.py → demo_spy_trade.py
    ├── trade_trajectory.py
    ├── trajectory_reward.py
    ├── regime_classifier.py
    ├── logistics_driver.py
    └── agentlightning_adapter.py
```

**Run:** `python demo_spy_trade.py`

### 3. Market-state observation stack

```
data_loader.py → canopyento_boundary_engine.py
    ├── B_s, E_i, rupture_pressure_score, regime_label
    ├── weekly stance vector (direction vs trade permission)
    ├── observer differential: T_a, R_o, T_v, observer_profile
    └── outputs/canopyento_boundary_{TICKER}.csv + .md + .json
              ↓
    capillary_engine.py (reads CanopyEnto CSV)
    ├── brownian_noise, wave_persistence, compression, surface_tension
    └── outputs/capillary_engine_{TICKER}.csv + .md + .json

pressure_field_dashboard.py (parallel — reads stock + option CSV directly)
    ├── MACD, RSI, CVD proxy, Volume, VWAP, gamma flip
    ├── LRP + rate-of-change derivatives + alerts
    ├── merges CanopyEnto boundary + observer metrics
    └── outputs/pressure_field_*.{html,csv,md,json}
```

**Run stack:**

```powershell
python canopyento_boundary_engine.py --ticker SPY
python capillary_engine.py --ticker SPY
python pressure_field_dashboard.py --ticker SPY --open
python laser_falcon_primary_engine.py --ticker SPCX --benchmark SPY --projection-days 30
streamlit run streamlit_laser_falcon.py
```

---

## Key metrics glossary

### Sacred derived metrics (TRPR ontology)

| Key | Name | Range | Meaning |
|-----|------|-------|---------|
| LRP | Latent Rupture Potential | [0, 1] | **Canonical** pre-rupture pressure signal |
| LRP_adjusted | LRP Adjusted (experimental) | [0, 1] | Restoration-adjusted sibling — **not canonical** |
| IDI | Inward Drift Index | [0, 1] | Inward collapse through coupling/dependency |
| R_o | Observer resolution | [0, 1] | Ability to reconstruct latent packet movement |
| T_v | Visibility horizon | sessions | Sessions before state becomes hard to resolve |
| T_a | Transition acceleration | [-1, 1] | Second derivative of pressure-state transition |
| F_r | Restoring field strength | [0, 1] | Pull back toward equilibrium |
| D_c | Dissipation capacity | [0, 1] | Absorb/release pressure without rupture |
| A_micro | Oscillation amplitude (local) | [0, 1] | Maps to sacred `A_f`; not attractor field |
| C_w | Capillary wave score | [0, 1] | Amplitude × persistence instability |

### Named field regimes (sacred ontology `named_regimes`)

`RESTORED_EQUILIBRIUM` · `ACTIVE_COMPENSATION` · `WEAKENING_RESTORATION` · `CAPILLARY_PRE_RUPTURE` · `ENTROPIC_DEGRADATION`

Assigned by `field_regime_engine.py` from physics metrics. Map inward only — definitions live in `TRPR/ontology/packet_ontology.yaml`.

### LRP regimes (from sacred ontology)

| Regime | Range |
|--------|-------|
| STABLE | [0.0, 0.30) |
| PRESSURE_BUILDING | [0.30, 0.60) |
| PRE_RUPTURE | [0.60, 0.85) |
| RUPTURE_IMMINENT | [0.85, 1.0] |

### LRP implementation (finance-crow engine)

Implemented in `pressure_field_derivatives.py`. **Not** a simple linear ratio — uses weighted raw score + sigmoid to avoid saturation:

- Weights: T_a_norm (0.20), observer blindspot (0.20), gamma (0.20), VWAP (0.15), CVD (0.10), MACD (0.15)
- Mild absorption dampening, then `LRP = sigmoid(4 × (LRP_raw − 0.5))`
- Debug payload: `outputs/lrp_debug_{TICKER}.json`
- **Loop closure:** `LRP_adjusted` applies restoration/capillary/hysteresis/observer multipliers to `LRP_raw` — experimental, compare against baseline `LRP`

### JSON snapshot key ordering

`pressure_field_schema.py` preserves **core keys first** (pre-loop-closure order), then **extended keys** appended (`LRP_adjusted`, physics fields). Downstream scripts keyed on core fields remain stable.

### CanopyEnto boundary metrics

| Metric | Meaning |
|--------|---------|
| B_s | Boundary stress frequency |
| E_i | Energy injection rate |
| rupture_pressure_score | B_s × E_i |

**Boundary regimes:** IDLE, PRESSURE_BUILDING, ENERGY_INJECTION, CONTAINMENT_STRESS, RUPTURE_CANDIDATE

### Observer differential (CanopyEnto)

| Variable | Meaning |
|----------|---------|
| T_a | d²P/dt² — transitional acceleration of rupture pressure |
| R_o | Observational resolution (0–1); higher = earlier detection |
| T_v | Visibility horizon (0–10 sessions); T_v = f(R_o) |
| observer_profile | passenger / pilot / mechanic (from R_o tiers) |

### Capillary metrics

```
capillary_score = (brownian_noise × wave_persistence × compression) / max(surface_tension, 0.05)
```

**Regimes:** ABSORBING_NOISE, CRUISE_SURFACE_ACTIVE, SURFACE_RIPPLING, PINCH_OFF_WATCH, CAPILLARY_RUPTURE_RISK

### Stable JSON snapshot keys

`pressure_field_schema.py` defines `STABLE_SNAPSHOT_KEYS` shared by `pressure_field_latest_{TICKER}.json` and `canopyento_weekly_stance_{TICKER}.json`:

`ticker`, `timestamp`, `close`, `macd_regime`, `rsi_regime`, `cvd_regime`, `volume_ratio`, `vwap_distance_pct`, `gamma_flip`, `gamma_flip_distance_pct`, `canopy_regime`, `T_a`, `T_a_norm`, `T_a_regime`, `R_o`, `T_v`, `observer_profile`, `LRP`, `LRP_regime`, `d_canopy_pressure`, `dd_canopy_pressure`, `d_R_o`, `d_T_v`, `d_gamma_flip_distance`, `d_vwap_distance`

---

## Common tasks for LLMs

| Task | Start here |
|------|------------|
| Fix LRP false positives / calibration | `pressure_field_derivatives.py`, `tests/test_pressure_field.py`, `outputs/lrp_debug_{TICKER}.json` |
| Add a new market sensor to dashboard | `pressure_field_dashboard.py` → map in `config/pressure_ontology.yaml` (not sacred YAML) |
| Add restoration/capillary physics | `restoration_field_engine.py`, `capillary_wave_engine.py`, wire via `pressure_field_physics.py` |
| Add a sacred primitive | `TRPR/ontology/packet_ontology.yaml` + bump `schema_version` + update charter + tests |
| Extend observer metrics | `canopyento_boundary_engine.py`, then wire into `pressure_field_schema.py` |
| Add dashboard field to JSON snapshot | `pressure_field_schema.py` `STABLE_SNAPSHOT_KEYS` + dashboard writer |
| Understand trade failure scoring | `recursive_trade_failure.py`, `recursive_weight_engine.py` |
| Run all tests | `python -m unittest tests.test_pressure_field tests.test_packet_ontology -v` |

---

## Constraints and pitfalls

1. **Not live trading** — no broker API, no order execution; CSV replay only
2. **CVD is a proxy** — signed-volume (close direction × volume), not tick-true cumulative volume delta
3. **Missing option chain is OK** — gamma fields become `null`, `gamma_regime: NO_CHAIN`; dashboard still runs
4. **Do not redefine ontology primitives** in engine code or `config/` — map inward only
5. **Do not hardwire all engines** to `packet_ontology_loader.py` until deliberate integration pass
6. **NaN/inf handling** — use `pressure_field_schema.safe_float()` for JSON snapshots; first derivative row defaults to 0.0
7. **Windows paths** — default data at `F:/inputs/...`; override with `--stock-dir` / `--option-dir`
8. **Prefer `outputs/`** over root-level `recursive_weights_*.json` legacy copies

---

## Dependencies

- Python 3.11+
- `numpy`, `pandas`, `matplotlib` (visualizer), `pyyaml` (TRPR ontology loader)
- Optional: `ffmpeg` (MP4/GIF export), `pygame` (`disposition_ring_toss.py`)

---

## Tests

```powershell
python -m unittest tests.test_pressure_field tests.test_packet_ontology tests.test_pressure_field_physics tests.test_lrp_loop_closure tests.test_elastic_rebound tests.test_laser_falcon tests.test_chain_integrity tests.test_chain_compatibility -v
```

Expected: **48+ tests**.

---

## Related docs

| File | Audience | Purpose |
|------|----------|---------|
| [`README.md`](README.md) | Humans | CLI flags, module reference, quick start |
| [`summary.md`](summary.md) | LLMs / agents | Architecture, file tree, sacred rules, glossary |
| [`archive/log.md`](archive/log.md) | Both | Conversation milestones |
| [`archive/transition_framework.md`](archive/transition_framework.md) | Both | Universal transition framework + A_d frontier |
| [`archive/transition_framework.tex`](archive/transition_framework.tex) | Both | Formal equations (LaTeX) |
| [`TRPR/ontology/ONTOLOGY_CHARTER.md`](TRPR/ontology/ONTOLOGY_CHARTER.md) | Both | Ontology drift prevention |
| [`TRPR/ontology/packet_ontology.yaml`](TRPR/ontology/packet_ontology.yaml) | Both | Sacred vocabulary (schema v0.1.0) |
| [`config/pressure_ontology.yaml`](config/pressure_ontology.yaml) | Both | Market domain mapping layer |
