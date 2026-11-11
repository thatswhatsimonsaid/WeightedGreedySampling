# Controls v2

## Change in area vs passive, mean over 13 datasets

| Selector | First 25% | Full budget | better / worse than passive (25%) | better / worse than passive (full) |
|---|---|---|---|---|
| MAB (std) | -6.20% | -2.36% | 9 / 2 | 9 / 2 |
| RandomArm (v2) | -6.16% | -2.33% | 9 / 2 | 9 / 2 |
| Fixed w=0.00 | -3.97% | -1.20% | 6 / 2 | 4 / 2 |
| Fixed w=0.25 | -5.95% | -2.23% | 10 / 2 | 10 / 2 |
| Fixed w=0.50 | -6.21% | -2.49% | 10 / 2 | 10 / 2 |
| Fixed w=0.75 | -7.15% | -2.84% | 10 / 2 | 8 / 2 |
| Fixed w=1.00 | -5.46% | -2.05% | 9 / 1 | 8 / 2 |
| MAB (c=2.0), September | -6.27% | -2.37% | 9 / 2 | 8 / 2 |
| RandomArm, September | -6.16% | -2.33% | 9 / 2 | 9 / 2 |

## MAB (std) vs RandomArm (v2), paired per dataset, 95% CI over seeds (% of reference area; negative = MAB lower)

|  | First 25% | Full budget |
|---|---|---|
| better | 0 | 0 |
| worse | 0 | 0 |
| tie | 13 | 13 |

| Dataset | First 25% |  | Full budget |  |
|---|---|---|---|---|
| beer | +0.39% [-0.70%, +1.50%] | tie | +0.09% [-0.34%, +0.51%] | tie |
| bodyfat | -0.18% [-1.72%, +1.36%] | tie | -0.18% [-0.83%, +0.46%] | tie |
| concrete_cs | +0.41% [-2.07%, +2.86%] | tie | +0.31% [-0.85%, +1.51%] | tie |
| concrete_flow | -1.75% [-4.25%, +0.78%] | tie | -0.51% [-1.60%, +0.56%] | tie |
| concrete_slump | -1.32% [-4.15%, +1.52%] | tie | -0.62% [-1.88%, +0.63%] | tie |
| dgp_three_regime | +0.43% [-0.11%, +0.96%] | tie | +0.13% [-0.03%, +0.28%] | tie |
| dgp_two_regime | +0.04% [-0.38%, +0.48%] | tie | +0.02% [-0.11%, +0.16%] | tie |
| housing | +0.51% [-0.67%, +1.68%] | tie | +0.06% [-0.39%, +0.51%] | tie |
| mpg | +0.27% [-0.81%, +1.37%] | tie | +0.08% [-0.29%, +0.45%] | tie |
| no2 | +0.33% [-0.60%, +1.22%] | tie | +0.06% [-0.28%, +0.39%] | tie |
| pm10 | +0.41% [-0.35%, +1.13%] | tie | +0.25% [-0.03%, +0.52%] | tie |
| qsar | +0.18% [-0.48%, +0.84%] | tie | +0.05% [-0.21%, +0.33%] | tie |
| yacht | -0.43% [-1.94%, +1.02%] | tie | -0.22% [-0.80%, +0.34%] | tie |

## MAB (std) vs best fixed weight, paired per dataset, 95% CI over seeds (best = lowest mean area among the five at that budget; % of reference area; negative = MAB lower)

|  | First 25% | Full budget |
|---|---|---|
| better | 0 | 0 |
| worse | 10 | 9 |
| tie | 3 | 4 |

| Dataset | Best fixed (25%) | First 25% |  | Best fixed (full) | Full budget |  |
|---|---|---|---|---|---|---|
| beer | Fixed w=0.25 | +1.31% [+0.22%, +2.41%] | worse | Fixed w=0.25 | +0.54% [+0.12%, +0.97%] | worse |
| bodyfat | Fixed w=0.75 | +1.38% [-0.10%, +2.90%] | tie | Fixed w=0.75 | +0.65% [+0.11%, +1.21%] | worse |
| concrete_cs | Fixed w=1.00 | +7.66% [+4.70%, +10.62%] | worse | Fixed w=1.00 | +3.87% [+2.57%, +5.17%] | worse |
| concrete_flow | Fixed w=1.00 | +5.80% [+2.71%, +8.98%] | worse | Fixed w=1.00 | +2.11% [+0.74%, +3.53%] | worse |
| concrete_slump | Fixed w=1.00 | +6.94% [+3.85%, +10.07%] | worse | Fixed w=1.00 | +1.62% [+0.23%, +3.01%] | worse |
| dgp_three_regime | Fixed w=1.00 | +2.18% [+1.58%, +2.77%] | worse | Fixed w=0.75 | -0.03% [-0.19%, +0.14%] | tie |
| dgp_two_regime | Fixed w=1.00 | +1.35% [+0.99%, +1.72%] | worse | Fixed w=1.00 | +0.42% [+0.29%, +0.55%] | worse |
| housing | Fixed w=0.25 | +2.51% [+1.07%, +4.01%] | worse | Fixed w=0.25 | +0.70% [+0.18%, +1.24%] | worse |
| mpg | Fixed w=0.75 | +2.86% [+1.78%, +3.87%] | worse | Fixed w=0.75 | +0.98% [+0.63%, +1.32%] | worse |
| no2 | Fixed w=0.50 | +0.98% [+0.12%, +1.79%] | worse | Fixed w=0.50 | +0.19% [-0.12%, +0.50%] | tie |
| pm10 | Fixed w=0.75 | +0.73% [-0.06%, +1.53%] | tie | Fixed w=0.00 | +0.47% [-0.02%, +0.95%] | tie |
| qsar | Fixed w=0.75 | +0.62% [-0.35%, +1.48%] | tie | Fixed w=0.25 | +0.22% [-0.08%, +0.52%] | tie |
| yacht | Fixed w=0.25 | +1.44% [+0.02%, +2.85%] | worse | Fixed w=0.25 | +0.78% [+0.18%, +1.40%] | worse |

## MAB (std): selector state, median over seeds

| Dataset | Seeds | Final pulls, w=0.25 | Final pulls, w=0.5 | Final pulls, w=0.75 | Bonus / arm-value spread at round 25 |
|---|---|---|---|---|---|
| beer | 100 | 112.5 | 107 | 110 | 1.66 |
| bodyfat | 100 | 71 | 71.5 | 74 | 1.52 |
| concrete_cs | 100 | 25 | 28.5 | 35.5 | 1.81 |
| concrete_flow | 100 | 25 | 25 | 40 | 1.74 |
| concrete_slump | 100 | 26.5 | 24 | 41 | 2.09 |
| dgp_three_regime | 100 | 820.5 | 440 | 6.5 | 0.652 |
| dgp_two_regime | 100 | 561 | 286.5 | 16 | 0.928 |
| housing | 100 | 149 | 149 | 163 | 1.49 |
| mpg | 100 | 121.5 | 96.5 | 127 | 1.48 |
| no2 | 100 | 159.5 | 147.5 | 122 | 1.67 |
| pm10 | 100 | 182.5 | 137 | 129.5 | 2.1 |
| qsar | 100 | 130 | 182 | 175 | 1.31 |
| yacht | 100 | 98 | 89.5 | 95 | 1.98 |
