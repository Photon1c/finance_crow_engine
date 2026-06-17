# Laser Falcon Summary — SPCX

Generated: 2026-06-17T17:06:00Z

## Data Health
- Status: **SPARSE**
- Contracts: 14
- Expirations: 2
- IV coverage: 85.71%

## Chain Integrity
- Status: **DEGRADED**
- Health score: 0.6171
- Blank expirations: 0
- Missing IV ratio: 0.1429
- Wide spread ratio: 0.0
- Duplicate contracts: 0

- Chain warning: Minimum strikes per expiration is 1

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
- Gamma compression: 0.0078
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

## Temporal Chain Differential
- Status: INSUFFICIENT
- Prior date: n/a
- Current date: n/a
- Pressure direction: **n/a**
- delta ATM IV: n/a
- delta call wing IV: n/a
- delta put wing IV: n/a
- delta dealer stress: n/a
- delta skew asymmetry: n/a
- Compatibility: n/a
- Contract universe drift: False

## IV Surface
- Status: SKIPPED
- Reason: Insufficient strike density for surface

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
- boundary_stress_proxy (B_s): 0.0078
- rupture_pressure_contributor: 0.016
- lrp_contributor: 0.6901

## Benchmark: SPY
- ATM IV delta vs SPY: 155.185

## Artifacts
- iv_skew: `outputs\laser_falcon\SPCX_iv_skew.png`
- iv_surface: `outputs\laser_falcon\SPCX_iv_surface.png`
- ou_iv_projection: `outputs\laser_falcon\SPCX_ou_iv_projection.png`
- stochastic_vol_projection: `outputs\laser_falcon\SPCX_stochastic_vol_projection.png`
