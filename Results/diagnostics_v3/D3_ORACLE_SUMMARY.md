# Diagnostic 3: ORACLE-reward bandit

## Change in area vs passive, mean over 13 datasets

| Selector | First 25% | Full budget |
|---|---|---|
| MAB (ORACLE test reward) | -6.47% | -2.50% |
| MAB (std) | -6.20% | -2.36% |
| RandomArm (v2) | -6.16% | -2.33% |
| Best fixed weight | -8.66% | -3.27% |

## Paired counts per dataset, 95% CI over seeds (better = lower area)

| Comparison |  | First 25% | Full budget |
|---|---|---|---|
| MAB (ORACLE test reward) vs RandomArm (v2) | better | 2 | 1 |
| MAB (ORACLE test reward) vs RandomArm (v2) | worse | 0 | 0 |
| MAB (ORACLE test reward) vs RandomArm (v2) | tie | 11 | 12 |
| MAB (ORACLE test reward) vs best fixed weight | better | 0 | 1 |
| MAB (ORACLE test reward) vs best fixed weight | worse | 9 | 7 |
| MAB (ORACLE test reward) vs best fixed weight | tie | 4 | 5 |

## MAB (ORACLE test reward): selector state, median over seeds

| Dataset | Seeds | Final pulls, w=0.25 | Final pulls, w=0.5 | Final pulls, w=0.75 | Most-pulled arm | Best fixed (25%) | Best fixed (full) |
|---|---|---|---|---|---|---|---|
| beer | 100 | 133.5 | 90.5 | 88 | w=0.25 | Fixed w=0.25 | Fixed w=0.25 |
| bodyfat | 100 | 69 | 68 | 81 | w=0.75 | Fixed w=0.75 | Fixed w=0.75 |
| concrete_cs | 100 | 25.5 | 26 | 26.5 | w=0.75 | Fixed w=1.00 | Fixed w=1.00 |
| concrete_flow | 100 | 27 | 26 | 35 | w=0.75 | Fixed w=1.00 | Fixed w=1.00 |
| concrete_slump | 100 | 26 | 31 | 29 | w=0.5 | Fixed w=1.00 | Fixed w=1.00 |
| dgp_three_regime | 100 | 392 | 263.5 | 715.5 | w=0.75 | Fixed w=1.00 | Fixed w=0.75 |
| dgp_two_regime | 100 | 324.5 | 213 | 327.5 | w=0.75 | Fixed w=1.00 | Fixed w=1.00 |
| housing | 100 | 152.5 | 154 | 132.5 | w=0.5 | Fixed w=0.25 | Fixed w=0.25 |
| mpg | 100 | 100.5 | 106.5 | 144 | w=0.75 | Fixed w=0.75 | Fixed w=0.75 |
| no2 | 100 | 173 | 152 | 117.5 | w=0.25 | Fixed w=0.50 | Fixed w=0.50 |
| pm10 | 100 | 170 | 125 | 136 | w=0.25 | Fixed w=0.75 | Fixed w=0.00 |
| qsar | 100 | 170.5 | 140 | 170 | w=0.25 | Fixed w=0.75 | Fixed w=0.25 |
| yacht | 100 | 72.5 | 99.5 | 85 | w=0.5 | Fixed w=0.25 | Fixed w=0.25 |
