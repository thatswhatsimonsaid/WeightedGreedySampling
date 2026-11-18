# Diagnostic 1: full-pool error with labelled points excluded

## Reproduction of the September curves (exact match / runs compared)

| Selector | Full-pool (paper) vs `paper_metric_fullpool` | Test RMSE vs `paper_protocol` |
|---|---|---|
| RandomArm | 0 / 1400 | 0 / 1300 |
| MAB (c=2.0) | 0 / 1400 | 0 / 1300 |
| Fixed w=0.00 | 0 / 1400 | 0 / 1300 |
| Fixed w=0.25 | 0 / 1400 | 0 / 1300 |
| Fixed w=0.50 | 0 / 1400 | 0 / 1300 |
| Fixed w=0.75 | 0 / 1400 | 0 / 1300 |
| Fixed w=1.00 | 0 / 1400 | 0 / 1300 |
| iGS | 0 / 1400 | 0 / 1300 |
| Passive | 0 / 1400 | 0 / 1300 |

## Change in area vs passive, mean over 14 datasets

| Selector | Full-pool (paper), first 25% | Full-pool (paper), full | Full-pool excl. labelled, first 25% | Full-pool excl. labelled, full | Test RMSE, first 25% | Test RMSE, full |
|---|---|---|---|---|---|---|
| RandomArm | -8.06% | -14.07% | -8.10% | -16.77% | -4.15% | -1.79% |
| MAB (c=2.0) | -7.79% | -13.90% | -7.84% | -16.67% | -4.01% | -1.77% |
| Fixed w=0.00 | -3.19% | -4.01% | -3.16% | -5.13% | -2.53% | -0.80% |
| Fixed w=0.25 | -6.05% | -11.73% | -6.08% | -14.53% | -3.42% | -1.51% |
| Fixed w=0.50 | -7.91% | -14.30% | -7.95% | -17.00% | -4.20% | -1.86% |
| Fixed w=0.75 | -9.19% | -15.00% | -9.23% | -17.57% | -4.68% | -2.08% |
| Fixed w=1.00 | -5.81% | -6.72% | -5.78% | -7.26% | -4.10% | -1.74% |
| iGS | -6.96% | -11.43% | -6.99% | -13.42% | -3.99% | -1.51% |

## Paired counts per dataset, 95% CI over seeds (better = lower area)

| Comparison | Metric | Budget | better | worse | tie |
|---|---|---|---|---|---|
| MAB (c=2.0) vs iGS | Full-pool (paper) | first 25% | 3 | 3 | 8 |
| MAB (c=2.0) vs iGS | Full-pool (paper) | full | 9 | 4 | 1 |
| MAB (c=2.0) vs iGS | Full-pool excl. labelled | first 25% | 4 | 3 | 7 |
| MAB (c=2.0) vs iGS | Full-pool excl. labelled | full | 9 | 4 | 1 |
| MAB (c=2.0) vs RandomArm | Full-pool (paper) | first 25% | 0 | 1 | 13 |
| MAB (c=2.0) vs RandomArm | Full-pool (paper) | full | 0 | 3 | 11 |
| MAB (c=2.0) vs RandomArm | Full-pool excl. labelled | first 25% | 0 | 1 | 13 |
| MAB (c=2.0) vs RandomArm | Full-pool excl. labelled | full | 0 | 2 | 12 |

## Two-Regime: relative AUC vs iGS (mean area / mean iGS area) and change vs passive

| Dataset | Selector | Full-pool (paper), 25%, rel. iGS | Full-pool (paper), 25%, vs passive | Full-pool (paper), full, rel. iGS | Full-pool (paper), full, vs passive | Full-pool excl. labelled, 25%, rel. iGS | Full-pool excl. labelled, 25%, vs passive | Full-pool excl. labelled, full, rel. iGS | Full-pool excl. labelled, full, vs passive | Test RMSE, 25%, rel. iGS | Test RMSE, 25%, vs passive | Test RMSE, full, rel. iGS | Test RMSE, full, vs passive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dgp_two_regime | MAB (c=2.0) | 1.0095 | -7.69% | 0.9563 | -28.80% | 1.0095 | -7.90% | 0.9318 | -35.90% | 1.0062 | +0.23% | 1.0009 | +0.10% |
| dgp_two_regime | Fixed w=0.25 | 1.0818 | -1.08% | 1.0969 | -18.33% | 1.0837 | -1.13% | 1.0739 | -26.12% | 1.0069 | +0.29% | 1.0016 | +0.16% |
| dgp_two_regime_gmm | MAB (c=2.0) | 1.0054 | -1.29% | 0.9993 | -3.51% | 1.0052 | -1.35% | 1.0029 | -3.52% | 1.0100 | +0.93% | 1.0027 | +0.22% |
| dgp_two_regime_gmm | Fixed w=0.25 | 1.0382 | +1.93% | 1.0443 | +0.83% | 1.0387 | +1.94% | 1.0402 | +0.07% | 1.0150 | +1.43% | 1.0057 | +0.52% |
