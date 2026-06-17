# Laser Falcon Summary — SPCX

Generated: 2026-06-17T05:52:07Z

## Data Health
- Status: **SPARSE**
- Contracts: 14
- Expirations: 2
- IV coverage: 85.71%


## IV Skew
- Expiration: Thu Jun 18 2026
- ATM IV: 168.525
- Put wing IV: None
- Call wing IV: 175.97
- Skew slope: 0.312686
- Put fear flag: False
- Call FOMO flag: False

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
- skew_instability_score: 0.0252
- surface_dislocation_score: 0.55
- vol_reversion_pressure: 0.0433
- option_liquidity_risk: 0.0572

## Benchmark: SPY
- ATM IV delta vs SPY: 155.185

## Artifacts
- iv_skew: `outputs\laser_falcon\SPCX_iv_skew.png`
- iv_surface: `outputs\laser_falcon\SPCX_iv_surface.png`
- ou_iv_projection: `outputs\laser_falcon\SPCX_ou_iv_projection.png`
- stochastic_vol_projection: `outputs\laser_falcon\SPCX_stochastic_vol_projection.png`
