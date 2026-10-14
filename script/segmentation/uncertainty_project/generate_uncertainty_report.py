import argparse
import csv
import os
import statistics


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values):
    return statistics.fmean(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report from boundary comparison CSV")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to sample_metrics.csv")
    parser.add_argument("--output_md", type=str, required=True, help="Output markdown report path")
    parser.add_argument("--threshold_hd95", type=float, default=0.5)
    parser.add_argument("--threshold_sd", type=float, default=0.1)
    parser.add_argument("--threshold_dice", type=float, default=0.0)
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--best_name", type=str, default="best_model")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--hard_split_percentile", type=float, default=50.0)
    args = parser.parse_args()

    rows = []
    with open(args.csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "sample_idx": int(row["sample_idx"]),
                "dice_baseline": safe_float(row.get("dice_baseline", 0.0)),
                "dice_uncertainty": safe_float(row.get("dice_uncertainty", 0.0)),
                "dice_diff": safe_float(row.get("dice_diff", 0.0)),
                "hd95_baseline": safe_float(row.get("hd95_baseline")),
                "hd95_uncertainty": safe_float(row.get("hd95_uncertainty")),
                "hd95_diff": safe_float(row.get("hd95_diff", 0.0)),
                "sd_baseline": safe_float(row.get("sd_baseline")),
                "sd_uncertainty": safe_float(row.get("sd_uncertainty")),
                "sd_diff": safe_float(row.get("sd_diff", 0.0)),
                "dice_ensemble": safe_float(row.get("dice_ensemble", row.get("dice_uncertainty", 0.0))),
                "hd95_ensemble": safe_float(row.get("hd95_ensemble", row.get("hd95_uncertainty", 0.0))),
                "sd_ensemble": safe_float(row.get("sd_ensemble", row.get("sd_uncertainty", 0.0))),
                "hd95_diff_ensemble": safe_float(row.get("hd95_diff_ensemble", 0.0)),
                "dice_diff_ensemble": safe_float(row.get("dice_diff_ensemble", 0.0)),
            })

    if not rows:
        raise ValueError(f"No rows found in CSV: {args.csv_path}")

    n = len(rows)
    methods = {
        "Baseline": {"dice": [r["dice_baseline"] for r in rows], "hd95": [r["hd95_baseline"] for r in rows], "sd": [r["sd_baseline"] for r in rows]},
        "CUCL": {"dice": [r["dice_uncertainty"] for r in rows], "hd95": [r["hd95_uncertainty"] for r in rows], "sd": [r["sd_uncertainty"] for r in rows]},
        "Ensemble": {"dice": [r["dice_ensemble"] for r in rows], "hd95": [r["hd95_ensemble"] for r in rows], "sd": [r["sd_ensemble"] for r in rows]},
    }

    improved_dice = [r for r in rows if r["dice_diff"] > args.threshold_dice]
    degraded_dice = [r for r in rows if r["dice_diff"] < -args.threshold_dice]
    improved_hd95 = [r for r in rows if r["hd95_diff"] > args.threshold_hd95]
    degraded_hd95 = [r for r in rows if r["hd95_diff"] < -args.threshold_hd95]
    improved_sd = [r for r in rows if r["sd_diff"] > args.threshold_sd]
    degraded_sd = [r for r in rows if r["sd_diff"] < -args.threshold_sd]

    best_cases = sorted(rows, key=lambda x: x["hd95_diff"], reverse=True)[: args.top_k]
    worst_cases = sorted(rows, key=lambda x: x["hd95_diff"])[: args.top_k]

    hd95_baseline = methods["Baseline"]["hd95"]
    split_pct = min(max(args.hard_split_percentile, 0.0), 100.0)
    sorted_hd95 = sorted(hd95_baseline)
    split_idx = int((split_pct / 100.0) * (len(sorted_hd95) - 1))
    split_threshold = sorted_hd95[split_idx]
    easy_rows = [r for r in rows if r["hd95_baseline"] <= split_threshold]
    hard_rows = [r for r in rows if r["hd95_baseline"] > split_threshold]

    def subset_mean_improvement(subset, method_key):
        if not subset:
            return 0.0
        return safe_mean([s["hd95_baseline"] - s[method_key] for s in subset])

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Segmentation Uncertainty Comparison Report\n\n")
        f.write(f"- Baseline: **{args.baseline_name}**\\n")
        f.write(f"- Best model: **{args.best_name}**\\n")
        f.write(f"- Number of evaluated samples: **{n}**\\n")

        f.write("## Aggregate Metrics (All Methods)\n\n")
        f.write("| Method | Dice Mean (↑) | HD95 Mean (↓) | SD Mean (↓) | Mean HD95 Gain vs Baseline (↑) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        baseline_hd95_vals = methods["Baseline"]["hd95"]
        for method_name, metric_dict in methods.items():
            if method_name == "Baseline":
                hd95_gain = 0.0
            else:
                hd95_gain = safe_mean([b - v for b, v in zip(baseline_hd95_vals, metric_dict["hd95"])])
            f.write(f"| {method_name} | {safe_mean(metric_dict['dice']):.4f} | {safe_mean(metric_dict['hd95']):.4f} | {safe_mean(metric_dict['sd']):.4f} | {hd95_gain:.4f} |\\n")
        f.write("\n")

        f.write("## CUCL vs Baseline Breakdown\n\n")
        f.write(f"- Dice improved: **{len(improved_dice)} / {n}** ({100*len(improved_dice)/n:.1f}%)\\n")
        f.write(f"- Dice degraded: **{len(degraded_dice)} / {n}** ({100*len(degraded_dice)/n:.1f}%)\\n")
        f.write(f"- HD95 significantly improved: **{len(improved_hd95)} / {n}** ({100*len(improved_hd95)/n:.1f}%)\\n")
        f.write(f"- HD95 significantly degraded: **{len(degraded_hd95)} / {n}** ({100*len(degraded_hd95)/n:.1f}%)\\n")
        f.write(f"- SD significantly improved: **{len(improved_sd)} / {n}** ({100*len(improved_sd)/n:.1f}%)\\n")
        f.write(f"- SD significantly degraded: **{len(degraded_sd)} / {n}** ({100*len(degraded_sd)/n:.1f}%)\\n\\n")

        f.write("## Easy/Hard Split by Baseline HD95\n\n")
        f.write(f"- Split percentile: **{split_pct:.1f}%**\\n")
        f.write(f"- Baseline HD95 split threshold: **{split_threshold:.4f}**\\n")
        f.write(f"- Easy cases (baseline HD95 <= threshold): **{len(easy_rows)}**\\n")
        f.write(f"- Hard cases (baseline HD95 > threshold): **{len(hard_rows)}**\\n\\n")

        f.write("| Method | Easy HD95 Gain vs Baseline (↑) | Hard HD95 Gain vs Baseline (↑) |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| CUCL | {subset_mean_improvement(easy_rows, 'hd95_uncertainty'):.4f} | {subset_mean_improvement(hard_rows, 'hd95_uncertainty'):.4f} |\\n")
        f.write(f"| Ensemble | {subset_mean_improvement(easy_rows, 'hd95_ensemble'):.4f} | {subset_mean_improvement(hard_rows, 'hd95_ensemble'):.4f} |\\n\\n")

        f.write("## Top HD95 Improvements (CUCL vs Baseline)\n\n")
        f.write("| Rank | Sample | HD95 Baseline | HD95 CUCL | HD95 Diff | SD Baseline | SD CUCL | SD Diff |\\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, row in enumerate(best_cases, start=1):
            f.write(f"| {i} | {row['sample_idx']} | {row['hd95_baseline']:.4f} | {row['hd95_uncertainty']:.4f} | {row['hd95_diff']:.4f} | {row['sd_baseline']:.4f} | {row['sd_uncertainty']:.4f} | {row['sd_diff']:.4f} |\\n")

        f.write("\n## Top HD95 Degradations (CUCL vs Baseline)\n\n")
        f.write("| Rank | Sample | HD95 Baseline | HD95 CUCL | HD95 Diff | SD Baseline | SD CUCL | SD Diff |\\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, row in enumerate(worst_cases, start=1):
            f.write(f"| {i} | {row['sample_idx']} | {row['hd95_baseline']:.4f} | {row['hd95_uncertainty']:.4f} | {row['hd95_diff']:.4f} | {row['sd_baseline']:.4f} | {row['sd_uncertainty']:.4f} | {row['sd_diff']:.4f} |\\n")

    print(f"Report written: {args.output_md}")


if __name__ == "__main__":
    main()
