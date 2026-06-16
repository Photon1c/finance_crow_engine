# CanopyEnto Boundary Rupture Report

- **Ticker:** SPY
- **Lookback:** 20 sessions
- **Boundary tolerance:** 0.0030
- **Volume window:** 20 sessions
- **Weekly stance window:** 5 sessions

## Latest Snapshot

- **Latest date:** 2026-06-12
- **Latest close:** 741.75
- **Latest B_s:** 0.3000
- **Latest E_i:** 1.0526
- **Latest rupture_pressure_score:** 0.3158
- **Latest regime_label:** PRESSURE_BUILDING
- **Latest regime_persistence:** 1 sessions

## Weekly Stance Filter

A forecast should not only predict direction. It should estimate whether the observed system has finished becoming the thing being predicted. This engine forecasts **state maturity**, not price.

- **Direction bias:** bullish (`direction_score=0.03`)
- **Trade permission:** weak (`stance_confidence=+0.18`)
- **Stance quadrant:** bullish / unresolved
- **Gate stance:** LOW-CONFIDENCE CRUISE MODE
- **Regime persistence:** 1 consecutive sessions in `PRESSURE_BUILDING`

### Stance Vector

| Component | Score |
| :--- | ---: |
| direction_score | 0.03 |
| timing_score | 0.72 |
| magnitude_score | 0.61 |
| volatility_score | 0.21 |
| packet_completion_confidence | 0.64 |
| absorption_confidence | 0.77 |
| hidden_process_uncertainty | 0.59 |
| continuation_probability | 0.43 |
| regime_persistence | 1 |
| rupture_probability | 0.32 |

### Recommended Model Action

Bullish bias detected, but packet completion is low. Current state: bullish / unresolved. Wait for confirmation or reduce position size. The observed packet has not resolved enough to trust the directional conclusion.

## Boundary Model

Rupture probability increases when a constrained system is repeatedly stressed while continuing to absorb energy without release.

## Top 10 Rupture Pressure Scores

| Date | Close | B_s | E_i | Rupture Score | Regime | Stance |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| 2020-09-03 | 345.39 | 0.6000 | 2.6040 | 1.5624 | RUPTURE_CANDIDATE | bullish / unresolved |
| 2017-10-25 | 255.29 | 0.8500 | 1.7599 | 1.4959 | RUPTURE_CANDIDATE | bullish / unresolved |
| 2017-03-01 | 239.78 | 0.7500 | 1.9770 | 1.4828 | RUPTURE_CANDIDATE | bullish / actionable |
| 2017-08-10 | 243.76 | 0.7000 | 2.1127 | 1.4789 | RUPTURE_CANDIDATE | bearish / unresolved |
| 2017-10-20 | 257.11 | 0.9000 | 1.5690 | 1.4121 | RUPTURE_CANDIDATE | bullish / actionable |
| 2020-02-28 | 296.26 | 0.4000 | 3.4673 | 1.3869 | RUPTURE_CANDIDATE | bearish / actionable |
| 2024-12-20 | 591.15 | 0.5000 | 2.6766 | 1.3383 | RUPTURE_CANDIDATE | bearish / actionable |
| 2018-02-05 | 263.93 | 0.5000 | 2.6504 | 1.3252 | RUPTURE_CANDIDATE | bearish / actionable |
| 2018-02-06 | 269.13 | 0.4500 | 2.8256 | 1.2715 | RUPTURE_CANDIDATE | bearish / actionable |
| 2020-09-04 | 342.57 | 0.5500 | 2.2838 | 1.2561 | RUPTURE_CANDIDATE | bearish / actionable |
