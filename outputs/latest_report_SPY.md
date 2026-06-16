# RECURSIVE TRADE FAILURE AGGREGATOR - REPORT

> *AI-Intermediary Log File Format - Structured Memory Vector Initialization*

### Trade Packet 1: SPY Put
- **Current Status:** Near Breakeven
- **Primary Classified Failure:** `Timing`
- **Aggregate Failure Score:** `0.44` (0=Pass, 1=Total Failure)
- **Market Context (CSV):** Spot $737.05 (2026-06-09 00:00:00); chain 06_09_2026; optimal SPY260616P00735000 $735; IV 0.18; delta -0.45; DTE 7; 5d ret -2.96%; realized vol 12.84%

| Component | Score (-1 to +1) |
| :--- | :---: |
| Directional Thesis | +0.4 |
| Timing Thesis      | -0.5 |
| Magnitude Thesis   | -0.2 |
| Volatility Dynamics| -0.1 |
| Catalyst Quality   | +0.6 |
| Exit Discipline    | +0.6 |

**Model Update Actions:**
- *Suggested Parameter Adjustments:* Absorption d(+0.04), Pressure d(+0.06)
- *AI Log:* `CSV-derived optimal SPY260616P00735000: IV 0.18, delta -0.45`

---
### Most Similar Historical Packets: SPY Put

| Distance | Timestamp | Trade ID | Instrument | Failure | Failure Score | Weighted Score | Status |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| 0.000 | 2026-06-11T02:56:23+00:00 | 1 | SPY Put | Timing | 0.44 | 0.39 | Near Breakeven |
| 0.573 | 2026-06-10T03:35:03+00:00 | 1 | SPY Put | Timing | 0.43 | 0.37 | Near Breakeven |
| 0.573 | 2026-06-11T02:55:16+00:00 | 1 | SPY Put | Timing | 0.43 | 0.38 | Near Breakeven |
| 0.573 | 2026-06-11T02:56:04+00:00 | 1 | SPY Put | Timing | 0.43 | 0.38 | Near Breakeven |
| 0.942 | 2026-06-10T03:30:14+00:00 | 1 | SPY Put | Timing | 0.42 | 0.42 | Near Breakeven |
**Updated Theory:** Updated theory: increase `timing` weight because `Timing` recurred in 5/5 similar cases and `timing` penalty appeared in 5/5 of them.

### Trade Packet 2: UEC Weekly Call
- **Current Status:** -100% Premium
- **Primary Classified Failure:** `Direction`
- **Aggregate Failure Score:** `0.89` (0=Pass, 1=Total Failure)

| Component | Score (-1 to +1) |
| :--- | :---: |
| Directional Thesis | -1.0 |
| Timing Thesis      | -0.8 |
| Magnitude Thesis   | -1.0 |
| Volatility Dynamics| -0.9 |
| Catalyst Quality   | -1.0 |
| Exit Discipline    | +0.0 |

**Model Update Actions:**
- *Suggested Parameter Adjustments:* Absorption d(-0.50), Pressure d(+0.60)
- *AI Log:* `Catalyst inversion; trigger structural IV crush penalty`

---
### Most Similar Historical Packets: UEC Weekly Call

| Distance | Timestamp | Trade ID | Instrument | Failure | Failure Score | Weighted Score | Status |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| 0.000 | 2026-06-10T03:30:14+00:00 | 2 | UEC Weekly Call | Direction | 0.89 | 0.84 | -100% Premium |
| 0.000 | 2026-06-10T03:32:23+00:00 | 2 | UEC Weekly Call | Direction | 0.89 | 0.85 | -100% Premium |
| 0.000 | 2026-06-10T03:33:04+00:00 | 2 | UEC Weekly Call | Direction | 0.89 | 0.84 | -100% Premium |
| 0.000 | 2026-06-10T03:35:03+00:00 | 2 | UEC Weekly Call | Direction | 0.89 | 0.85 | -100% Premium |
| 0.000 | 2026-06-10T16:45:36+00:00 | 2 | UEC Weekly Call | Direction | 0.89 | 0.85 | -100% Premium |
**Updated Theory:** Updated theory: increase `direction` weight because `Direction` recurred in 5/5 similar cases and `direction` penalty appeared in 5/5 of them.

## Recursive Model State

## Model Weights After Update (SPY)

| Component | Weight |
| :--- | :---: |
| Direction | 1.131 |
| Timing | 1.336 |
| Magnitude | 1.253 |
| Volatility | 1.263 |
| Catalyst | 1.101 |
| Exit | 1.070 |
| Theta Risk | 1.268 |
| Cvd Confirmation | 1.000 |
## Model Weights After Update (UEC)

| Component | Weight |
| :--- | :---: |
| Direction | 1.450 |
| Timing | 1.405 |
| Magnitude | 1.450 |
| Volatility | 1.427 |
| Catalyst | 1.450 |
| Exit | 1.225 |
| Theta Risk | 1.180 |
| Cvd Confirmation | 1.450 |
