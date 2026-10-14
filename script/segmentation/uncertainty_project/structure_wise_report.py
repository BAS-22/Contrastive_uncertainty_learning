import argparse
import csv
import re
import statistics

from scipy.stats import wilcoxon


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def detect_structure_names(fieldnames):
    pattern = re.compile(r"^dice_baseline_(.+)$")
    names = []
    for f in fieldnames:
        m = pattern.match(f)
        if m:
            names.append(m.group(1))
    return names


def paired_values(rows, baseline_col, method_col):
    pairs = []
    for r in rows:
        b = safe_float(r.get(baseline_col))
        m = safe_float(r.get(method_col))
        if b is not None and m is not None:
            pairs.append((b, m))
    return pairs


def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def paired_wilcoxon(baseline_vals, method_vals):
    diffs = [m - b for b, m in zip(baseline_vals, method_vals)]
    if len(diffs) < 1 or all(d == 0 for d in diffs):
        return float("nan")
    try:
        _, p = wilcoxon(baseline_vals, method_vals)
        return p
    except ValueError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Per-structure (organ-wise) Dice/HD95/SD comparison from a sample_metrics.csv "
        "produced by engine.segment.test_with_comparison, with paired Wilcoxon significance tests."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to sample_metrics.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to write per-structure stats CSV")
    parser.add_argument("--output_md", type=str, required=True, help="Path to write per-structure markdown report")
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--method_name", type=str, default="CUL")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for Wilcoxon p-value")
    args = parser.parse_args()

    with open(args.csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in CSV: {args.csv_path}")

    structure_names = detect_structure_names(fieldnames)
    if not structure_names:
        raise ValueError(
            "No per-structure columns found (expected columns like 'dice_baseline_<structure>'). "
            "Re-run test_with_comparison with the per-class capture enabled."
        )

    metrics = ["dice", "hd95", "sd"]
    results = []
    for metric in metrics:
        for name in structure_names:
            baseline_col = f"{metric}_baseline_{name}"
            method_col = f"{metric}_uncertainty_{name}"
            pairs = paired_values(rows, baseline_col, method_col)
            n = len(pairs)
            baseline_vals = [b for b, _ in pairs]
            method_vals = [m for _, m in pairs]
            baseline_mean, baseline_std = mean_std(baseline_vals)
            method_mean, method_std = mean_std(method_vals)
            mean_diff = method_mean - baseline_mean
            p_value = paired_wilcoxon(baseline_vals, method_vals)
            results.append({
                "structure": name,
                "metric": metric,
                "n": n,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "method_mean": method_mean,
                "method_std": method_std,
                "mean_diff": mean_diff,
                "wilcoxon_p": p_value,
                "significant": (p_value == p_value) and p_value < args.alpha,
            })

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "structure", "metric", "n",
            "baseline_mean", "baseline_std",
            f"{args.method_name}_mean", f"{args.method_name}_std",
            "mean_diff", "wilcoxon_p", "significant",
        ])
        for r in results:
            writer.writerow([
                r["structure"], r["metric"], r["n"],
                r["baseline_mean"], r["baseline_std"],
                r["method_mean"], r["method_std"],
                r["mean_diff"], r["wilcoxon_p"], r["significant"],
            ])
    print(f"Saved per-structure stats CSV: {args.output_csv}")

    metric_titles = {"dice": "Dice (↑)", "hd95": "HD95 (↓)", "sd": "Surface Distance (↓)"}
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Structure-wise (Organ-wise) Comparison\n\n")
        f.write(f"- Baseline: **{args.baseline_name}**\n")
        f.write(f"- Method: **{args.method_name}**\n")
        f.write(f"- Significance threshold: p < {args.alpha} (paired Wilcoxon signed-rank test)\n\n")

        for metric in metrics:
            f.write(f"## {metric_titles.get(metric, metric)}\n\n")
            f.write(f"| Structure | n | {args.baseline_name} mean ± std | {args.method_name} mean ± std | Mean diff | Wilcoxon p | Significant |\n")
            f.write("|---|---:|---:|---:|---:|---:|:---:|\n")
            for r in [x for x in results if x["metric"] == metric]:
                sig_mark = "**Yes**" if r["significant"] else "No"
                p_str = f"{r['wilcoxon_p']:.4f}" if r["wilcoxon_p"] == r["wilcoxon_p"] else "n/a"
                f.write(
                    f"| {r['structure']} | {r['n']} | "
                    f"{r['baseline_mean']:.4f} ± {r['baseline_std']:.4f} | "
                    f"{r['method_mean']:.4f} ± {r['method_std']:.4f} | "
                    f"{r['mean_diff']:+.4f} | {p_str} | {sig_mark} |\n"
                )
            f.write("\n")

    print(f"Saved per-structure markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
