# Pressure Field Report — SPY

- **As of:** 2026-06-16
- **Chain date:** 06_15_2026
- **Close:** 750.33

## Latent Rupture Potential

- **LRP:** 0.3050
- **LRP_regime:** PRESSURE_BUILDING

## Restoration-Adjusted Rupture Risk — experimental (not canonical)

- **LRP Adjusted (experimental):** 0.2441
- **LRP_adjusted_regime (experimental):** STABLE
- **restoration_damper:** 0.5500
- **capillary_boost:** 1.1050
- **hysteresis_boost:** 1.1857
- **observer_boost:** 1.0259

_Baseline LRP = pressure-driven risk (canonical). LRP Adjusted (experimental) = compensatory-capacity-adjusted risk — do not treat as canonical._

_Baseline LRP = pressure signal; LRP_adjusted (experimental) = pressure after restoration/capillary/hysteresis/observer modifiers._

## Rate-of-Change Snapshot

- **d_canopy_pressure:** +0.031798
- **dd_canopy_pressure:** +0.014593
- **d_R_o:** -0.071821
- **d_T_v:** +0.700000
- **d_gamma_flip_distance:** -0.587953
- **d_vwap_distance:** -0.704842

## Derived Derivatives

| Metric | Latest |
| :--- | ---: |
| d_macd_pressure | +0.600385 |
| d_rsi_energy | -0.066928 |
| d_cvd_force | +0.071058 |
| d_volume_energy | +0.105993 |
| d_vwap_attractor_distance | -0.704842 |
| d_gamma_flip_distance | -0.587953 |
| d_canopy_pressure | +0.031798 |
| d_observability_R_o | -0.071821 |
| d_visibility_horizon_T_v | +0.700000 |
| dd_canopy_pressure | +0.014593 |
| dd_observability_R_o | -0.047575 |
| dd_coherence_proxy | -0.218081 |
| d_R_o | -0.071821 |
| d_T_v | +0.700000 |
| d_vwap_distance | -0.704842 |

## Rate-of-Change Alerts

- OBSERVABILITY_DECAY_ACCELERATING
- COHERENCE_LOSS_ACCELERATING
- GAMMA_BOUNDARY_APPROACHING
- PRESSURE_ACCELERATION_POSITIVE

## Restoration & Capillary Physics

- **F_r:** 0.7315
- **D_c / dissipation_score:** 0.4303
- **restoration_ratio:** 2.3981
- **A_micro:** 0.7269
- **C_w / capillary_wave_score:** 0.4198
- **field_regime:** 
- **entropy_score:** 0.6258

## Elastic Rebound & Hidden Reservoir

- **elastic_strain_score:** 0.0326
- **gamma_rebound_regime:** LOCKED_FAULT
- **hidden_reservoir_pressure:** 1.0000
- **pressure_relocation_ratio:** 0.1413
- **false_stability_flag:** 1
- **observability_gap_score:** 0.4897

- _Positive gamma may behave like a locked fault: surface volatility compresses while hidden positioning strain accumulates._
- _Reduced visible pressure is not always true dissipation; some pressure may be relocating into a hidden reservoir._
- _False stability is flagged when observable pressure falls while hidden strain or reservoir pressure rises._

## Gamma Flip

- **Flip strike:** 740.0
- **Distance %:** 1.3767275732011304
- **Regime:** ABOVE_FLIP_POSITIVE_GAMMA

## Observer Differential

- **T_a_regime:** CRUISE
- **R_o:** 0.5073
- **T_v:** 3.7
- **observer_profile:** pilot

_Previous session close: 754.83_

