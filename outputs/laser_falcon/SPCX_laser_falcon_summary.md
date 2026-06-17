# Laser Falcon Summary — SPCX

Generated: 2026-06-17T15:57:03Z

## Data Health
- Status: **SPARSE**
- Contracts: 14
- Expirations: 2
- IV coverage: 85.71%


## IV Skew
- Expiration: Thu Jun 18 2026
- ATM IV: 168.525
- Put wing IV: 166.99
- Call wing IV: 175.97
- Skew slope: 0.312686
- Skew ratio: 0.949
- Skew asymmetry: -0.0533
- Calls overpriced: False
- Puts overpriced: False
- Skew inversion: True

## Options Pressure Metrics
- Gamma compression: 0.0126
- Vol expansion (ATM/realized): None
- Skew asymmetry pressure: -0.0533
- Dealer hedging stress: 0.0
- 30d realized vol: None%

## Anomaly Detection
- Primary: **IPO_INSTABILITY**
- Labels: IPO_INSTABILITY, SKEW_INVERSION
- Severity: 1.0

## Vol Regime
- Regime: **HIGH_VOL_REGIME**
- Confidence: 0.7

## Vol Arbitrage / Dislocation
- Status: INSUFFICIENT
- Dislocation: n/a
- Potential dislocation: False

## IV Surface
- Status: SKIPPED
- Reason: single expiration only — 2D skew preferred

## OU IV Mean Reversion
- IV0: 1.6852500000000001
- Terminal mean: 1.612208295459245
- Half-life (days): 63.24968022609501

## Stochastic Volatility
- Terminal median price: 181.36064189778835
- Terminal median vol: 1.6816206272140257

## Pressure-Field Mapping (local)
- iv_pressure_score: 1.0
- skew_instability_score: 0.2359
- surface_dislocation_score: 0.55
- vol_reversion_pressure: 0.0433
- option_liquidity_risk: 0.0572
- energy_injection_proxy (E_i): 0.25
- boundary_stress_proxy (B_s): 0.0126
- rupture_pressure_contributor: 0.016
- lrp_contributor: 0.6901

## Benchmark: SPY
- ATM IV delta vs SPY: 155.185

## Artifacts
- iv_skew: `outputs\laser_falcon\SPCX_iv_skew.png`
- iv_surface: `outputs\laser_falcon\SPCX_iv_surface.png`
- ou_iv_projection: `outputs\laser_falcon\SPCX_ou_iv_projection.png`
- stochastic_vol_projection: `outputs\laser_falcon\SPCX_stochastic_vol_projection.png`
