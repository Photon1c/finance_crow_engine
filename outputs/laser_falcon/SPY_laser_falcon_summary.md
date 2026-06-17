# Laser Falcon Summary — SPY

Generated: 2026-06-17T15:57:05Z

## Data Health
- Status: **SPARSE**
- Contracts: 120
- Expirations: 10
- IV coverage: 97.5%


## IV Skew
- Expiration: Fri Jun 26 2026
- ATM IV: 13.34
- Put wing IV: 13.34
- Call wing IV: 13.14
- Skew slope: -0.195143
- Skew ratio: 1.0152
- Skew asymmetry: 0.015
- Calls overpriced: False
- Puts overpriced: False
- Skew inversion: False

## Options Pressure Metrics
- Gamma compression: 0.0309
- Vol expansion (ATM/realized): 0.9183
- Skew asymmetry pressure: 0.015
- Dealer hedging stress: 0.5305
- 30d realized vol: 14.5267%

## Anomaly Detection
- Primary: **NORMAL**
- Labels: NORMAL
- Severity: 0.1

## Vol Regime
- Regime: **LOW_VOL_REGIME**
- Confidence: 0.5

## Vol Arbitrage / Dislocation
- Status: SKIPPED
- Dislocation: n/a
- Potential dislocation: False

## IV Surface
- Status: OK
- Reason: ok

## OU IV Mean Reversion
- IV0: 0.1334
- Terminal mean: 0.1361937415344545
- Half-life (days): 63.24968022609501

## Stochastic Volatility
- Terminal median price: 759.1525048852628
- Terminal median vol: 0.10219847816287747

## Pressure-Field Mapping (local)
- iv_pressure_score: 0.1334
- skew_instability_score: 0.005
- surface_dislocation_score: 0.4
- vol_reversion_pressure: 0.0209
- option_liquidity_risk: 0.04
- energy_injection_proxy (E_i): 0.2296
- boundary_stress_proxy (B_s): 0.0309
- rupture_pressure_contributor: 0.2697
- lrp_contributor: 0.217

## Benchmark: SPY
- Benchmark comparison unavailable

## Artifacts
- iv_skew: `outputs\laser_falcon\SPY_iv_skew.png`
- iv_surface: `outputs\laser_falcon\SPY_iv_surface.png`
- ou_iv_projection: `outputs\laser_falcon\SPY_ou_iv_projection.png`
- stochastic_vol_projection: `outputs\laser_falcon\SPY_stochastic_vol_projection.png`
