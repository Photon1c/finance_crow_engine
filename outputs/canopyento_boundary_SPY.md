# CanopyEnto Boundary Rupture Report

- **Ticker:** SPY
- **Lookback:** 20 sessions
- **Boundary tolerance:** 0.0030
- **Volume window:** 20 sessions
- **Weekly stance window:** 5 sessions

## Latest Snapshot

- **Latest date:** 2026-06-15
- **Latest close:** 754.83
- **Latest B_s:** 0.3000
- **Latest E_i:** 1.1100
- **Latest rupture_pressure_score:** 0.3330
- **Latest regime_label:** PRESSURE_BUILDING
- **Latest regime_persistence:** 1 sessions

## Transitional Acceleration & Observer Differential

_Complex systems do not fail equally for all observers. Rupture becomes visible at different times depending on an observer's proximity to the system's internal pressure variables. The system changes before most observers are capable of perceiving the change._

- **T_a (transitional acceleration):** 0.264749
- **T_a_norm:** 0.6999
- **T_a_regime:** ACCELERATION_POSITIVE
- **R_o (observational resolution):** 0.5791
- **T_v (visibility horizon, sessions):** 3.0
- **Observer profile:** pilot

### T_a Regime Key

| Regime | Meaning |
| :--- | :--- |
| TAKEOFF_BEGINNING | Positive acceleration — thrust building |
| CRUISE | Near-zero acceleration — stable flight |
| THRUST_LOSS | Negative acceleration — momentum fading |
| DISSIPATION_CASCADE | Strong negative — energy bleeding out |

## Weekly Stance Filter

A forecast should not only predict direction. It should estimate whether the observed system has finished becoming the thing being predicted. This engine forecasts **state maturity**, not price.

- **Direction bias:** bullish (`direction_score=0.67`)
- **Trade permission:** actionable (`stance_confidence=+0.28`)
- **Stance quadrant:** bullish / actionable
- **Gate stance:** ACTIONABLE DIRECTIONAL STANCE
- **Regime persistence:** 1 consecutive sessions in `PRESSURE_BUILDING`

### Stance Vector

| Component | Score |
| :--- | ---: |
| direction_score | 0.67 |
| timing_score | 0.73 |
| magnitude_score | 0.60 |
| volatility_score | 0.24 |
| packet_completion_confidence | 0.64 |
| absorption_confidence | 0.58 |
| hidden_process_uncertainty | 0.53 |
| continuation_probability | 0.86 |
| regime_persistence | 1 |
| rupture_probability | 0.33 |

### Recommended Model Action

Actionable bullish stance. Packet completion and continuation probability support a directional position with normal sizing.

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
