"""
Levene and Brown-Forsythe tests for equality of HD95 variance.
Reviewer Weakness 1: the paper claims HD95 std reduction (8.14->4.88 mm) but
only reports a Wilcoxon test on means. This script formally tests the variance
claim so it can be reported in Table 2 / rebuttal.

Usage (login node, no GPU required):
  python hd95_variance_test.py
  python hd95_variance_test.py --csv /path/to/sample_metrics.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CSV = (
    "/SAN/medic/PosPretrain/project1/outputs/seg/uncertainty_reports/"
    "promis_profound_vit_zone_cucl_no_calibration_vs_baseline_799epoch_20260623_172447/"
    "comparison/sample_metrics.csv"
)

def run(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    baseline = df["hd95_baseline"].dropna().values
    method   = df["hd95_uncertainty"].dropna().values
    n = min(len(baseline), len(method))
    baseline, method = baseline[:n], method[:n]

    print(f"N = {n} samples")
    print(f"Baseline HD95 : mean={baseline.mean():.4f}  std={baseline.std(ddof=1):.4f}")
    print(f"Method   HD95 : mean={method.mean():.4f}  std={method.std(ddof=1):.4f}")
    print()

    # Levene (centre='mean') — classical Levene test
    lev_stat, lev_p = stats.levene(baseline, method, center="mean")
    print(f"Levene test (centre=mean):         F={lev_stat:.4f}  p={lev_p:.6f}")

    # Brown-Forsythe (centre='median') — more robust to non-normality
    bf_stat, bf_p = stats.levene(baseline, method, center="median")
    print(f"Brown-Forsythe test (centre=median): F={bf_stat:.4f}  p={bf_p:.6f}")

    print()
    alpha = 0.05
    sig = lev_p < alpha or bf_p < alpha
    print(f"Conclusion: variance reduction is {'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'} "
          f"(alpha={alpha})")

    # Also run Fligner-Killeen (nonparametric alternative)
    fk_stat, fk_p = stats.fligner(baseline, method)
    print(f"Fligner-Killeen (nonparametric):     chi2={fk_stat:.4f}  p={fk_p:.6f}")

    # Bartlett (assumes normality, for reference)
    bart_stat, bart_p = stats.bartlett(baseline, method)
    print(f"Bartlett test (assumes normality):   chi2={bart_stat:.4f}  p={bart_p:.6f}")

    print("\n--- For paper / rebuttal ---")
    print(f"HD95 variance: baseline {baseline.var(ddof=1):.2f} mm² vs method {method.var(ddof=1):.2f} mm²")
    print(f"Brown-Forsythe F={bf_stat:.3f}, p={bf_p:.4f} -> use this as primary test in rebuttal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    run(args.csv)
