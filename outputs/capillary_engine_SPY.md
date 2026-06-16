# Capillary Engine Report — SPY

- **Input:** `outputs\canopyento_boundary_SPY.csv`
- **As of:** 2026-06-12

## Latest Read

Market remains in cruise mode. Microscopic disturbances are being absorbed.

- **Capillary regime:** ABSORBING_NOISE
- **Capillary score:** 0.1692
- **Cruise integrity:** 0.5534
- **Pinch-off risk:** 0.2817

## Metric Table

| Metric | Value |
| :--- | ---: |
| close | 741.75 |
| returns | 0.005408 |
| brownian_noise | 0.6308 |
| wave_persistence | 0.4133 |
| compression | 0.2105 |
| surface_tension | 0.6661 |
| capillary_score | 0.1692 |
| cruise_integrity | 0.5534 |
| pinch_off_risk | 0.2817 |

## Regime Interpretation

| Capillary Score | Regime | Meaning |
| :--- | :--- | :--- |
| 0.00–0.25 | ABSORBING_NOISE | Microscopic disturbances dissipate cleanly |
| 0.25–0.45 | CRUISE_SURFACE_ACTIVE | Cruise surface intact with normal noise |
| 0.45–0.65 | SURFACE_RIPPLING | Disturbances begin to echo |
| 0.65–0.80 | PINCH_OFF_WATCH | Compression + persistence rising |
| 0.80–1.00 | CAPILLARY_RUPTURE_RISK | Brownian layer failing; pinch-off risk elevated |

**Current:** `ABSORBING_NOISE`

## CanopyEnto Integration

Richter / CanopyEnto detects stored pressure and boundary regime. Capillary Engine detects whether microscopic noise is still being absorbed. A market can remain in cruise mode while its Brownian layer becomes increasingly unstable.

| CanopyEnto Field | Value |
| :--- | :--- |
| regime | PRESSURE_BUILDING |
| rupture score | 0.3158 |
| rupture probability | 0.3158 |
| gate stance | LOW-CONFIDENCE CRUISE MODE |
| stance quadrant | bullish / unresolved |

## Notes / Caveats

- Capillary metrics are derived from local OHLCV and CanopyEnto CSV fields only.
- Short histories use reduced `min_periods`; early rows may be blank.
- `surface_tension` is a first-pass absorption proxy, not order-flow truth.
- High cruise integrity does not guarantee directional trade permission from CanopyEnto.
- Use Capillary read as a micro-stability overlay, not a standalone directional model.
