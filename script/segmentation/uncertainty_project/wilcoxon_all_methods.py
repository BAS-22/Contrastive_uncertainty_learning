r"""
Paired Wilcoxon signed-rank tests for all three methods vs baseline.
Covers aggregate (Dice / HD95 / SD) and per-structure SD.

Each method needs a sample_metrics.csv produced by compare_boundaries.py
(baseline vs that method).

Usage (CPU only):
  python script/segmentation/uncertainty_project/wilcoxon_all_methods.py \
      --cul_csv path/to/cul/comparison/sample_metrics.csv \
      --class_based_csv path/to/class_based/comparison/sample_metrics.csv \
      --classic_bl_csv path/to/classic_bl/comparison/sample_metrics.csv
  # or: --report_base <dir> to use the default report folder names below
"""
import argparse
import csv
import statistics
from pathlib import Path
from scipy.stats import wilcoxon

REPORT_BASE = Path(
    "/SAN/medic/PosPretrain/project1/outputs/seg/uncertainty_reports"
)

METHODS = {
    "CUL": REPORT_BASE
    / "promis_profound_vit_zone_cucl_no_calibration_vs_baseline_799epoch_20260623_172447"
    / "comparison/sample_metrics.csv",
    "Class-Based": REPORT_BASE
    / "promis_profound_vit_zone_cucl_class_based_vs_baseline_799epoch_20260629_165631"
    / "comparison/sample_metrics.csv",
    "Classic BL": REPORT_BASE
    / "promis_profound_vit_zone_classic_boundary_loss_vs_baseline_799epoch_20260629_210415"
    / "comparison/sample_metrics.csv",
}

STRUCTURES = [
    "bladder", "bone", "obturator_internus", "central_gland",
    "transition_zone", "rectum", "seminal_vesicle", "neurovascular_bundle",
]

STRUCT_DISPLAY = {
    "bladder": "Bladder",
    "bone": "Bone",
    "obturator_internus": "Obturator Int.",
    "central_gland": "Central Gland",
    "transition_zone": "Transition Zone",
    "rectum": "Rectum",
    "seminal_vesicle": "Seminal Vesicle",
    "neurovascular_bundle": "NVB",
}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def safe_float(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def paired_wilcoxon(rows, col_b, col_m):
    pairs = [(safe_float(r[col_b]), safe_float(r[col_m])) for r in rows]
    pairs = [(b, m) for b, m in pairs if b is not None and m is not None]
    if len(pairs) < 1:
        return float("nan"), float("nan"), float("nan"), 0
    b_vals = [b for b, _ in pairs]
    m_vals = [m for _, m in pairs]
    n = len(pairs)
    mean_b = statistics.fmean(b_vals)
    std_b = statistics.stdev(b_vals) if n > 1 else 0.0
    mean_m = statistics.fmean(m_vals)
    std_m = statistics.stdev(m_vals) if n > 1 else 0.0
    diffs = [m - b for b, m in pairs]
    if all(d == 0 for d in diffs):
        return mean_b, std_b, mean_m, std_m, float("nan"), n
    try:
        _, p = wilcoxon(b_vals, m_vals)
    except ValueError:
        p = float("nan")
    return mean_b, std_b, mean_m, std_m, p, n


def fmt_p(p):
    if p != p:
        return "  n/a  "
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def sig_star(p):
    if p != p:
        return " "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "   "


def run():
    data = {name: load_csv(path) for name, path in METHODS.items()}

    # ── Aggregate ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("AGGREGATE RESULTS (all structures pooled, paired Wilcoxon vs baseline)")
    print("=" * 70)

    for metric, col_b, col_m in [
        ("Dice",  "dice_baseline",  "dice_uncertainty"),
        ("HD95",  "hd95_baseline",  "hd95_uncertainty"),
        ("SD",    "sd_baseline",    "sd_uncertainty"),
    ]:
        print(f"\n  {metric}")
        print(f"  {'Method':<14} {'Baseline':>16}  {'Method':>16}  {'p-value':>10}  n")
        print(f"  {'-'*14} {'-'*16}  {'-'*16}  {'-'*10}  -")
        for name, rows in data.items():
            mb, sb, mm, sm, p, n = paired_wilcoxon(rows, col_b, col_m)
            print(
                f"  {name:<14} {mb:7.4f}±{sb:6.4f}  "
                f"{mm:7.4f}±{sm:6.4f}  {fmt_p(p):>10}  {n}"
            )

    # ── Per-structure SD ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PER-STRUCTURE SD (paired Wilcoxon vs baseline, α=0.05)")
    print("=" * 70)

    # header
    col_w = 15
    hdr = f"  {'Structure':<18}"
    for name in METHODS:
        hdr += f"  {name:>{col_w}}"
    print(hdr)
    print("  " + "-" * 18 + ("  " + "-" * col_w) * len(METHODS))

    sig_counts = {name: 0 for name in METHODS}
    baseline_shown = False

    for struct in STRUCTURES:
        col_b = f"sd_baseline_{struct}"
        col_m = f"sd_uncertainty_{struct}"
        row_str = f"  {STRUCT_DISPLAY[struct]:<18}"
        first = True
        for name, rows in data.items():
            mb, sb, mm, sm, p, n = paired_wilcoxon(rows, col_b, col_m)
            if not baseline_shown and first:
                # Print baseline once (same across all methods)
                pass
            star = sig_star(p)
            if p < 0.05:
                sig_counts[name] += 1
            row_str += f"  {mm:5.3f} {star} {fmt_p(p):>10}"
            first = False
        print(row_str)

    print("  " + "-" * 18 + ("  " + "-" * col_w) * len(METHODS))
    sig_row = f"  {'Sig. structures':<18}"
    for name in METHODS:
        sig_row += f"  {sig_counts[name]}/8{' ':>{col_w - 4}}"
    print(sig_row)

    # ── Baseline means for reference ──────────────────────────────────────
    print()
    print("BASELINE SD per structure (reference, from CUL CSV):")
    cul_rows = data["CUL"]
    for struct in STRUCTURES:
        pairs = [
            (safe_float(r[f"sd_baseline_{struct}"]),)
            for r in cul_rows
            if safe_float(r.get(f"sd_baseline_{struct}")) is not None
        ]
        vals = [p[0] for p in pairs]
        if vals:
            print(f"  {STRUCT_DISPLAY[struct]:<18}  {statistics.fmean(vals):.4f} ± {statistics.stdev(vals):.4f}")

    # ── Significance summary table ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("SIGNIFICANCE SUMMARY (SD, paired Wilcoxon, α=0.05)")
    print("=" * 70)
    print(f"  {'Structure':<20}  {'CUL':>10}  {'Class-Based':>12}  {'Classic BL':>12}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*12}")
    for struct in STRUCTURES:
        col_b = f"sd_baseline_{struct}"
        col_m = f"sd_uncertainty_{struct}"
        row_str = f"  {STRUCT_DISPLAY[struct]:<20}"
        for name, rows in data.items():
            *_, p, _ = paired_wilcoxon(rows, col_b, col_m)
            mark = f"✓ {fmt_p(p)}" if p < 0.05 else f"— {fmt_p(p)}"
            row_str += f"  {mark:>12}"
        print(row_str)
    print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*12}")
    sig_row = f"  {'Sig. total':<20}"
    for name in METHODS:
        sig_row += f"  {sig_counts[name]}/8{'':>6}"
    print(sig_row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report_base", type=str, default=None,
                        help="Folder containing the uncertainty_reports sub-folders")
    parser.add_argument("--cul_csv", type=str, default=None)
    parser.add_argument("--class_based_csv", type=str, default=None)
    parser.add_argument("--classic_bl_csv", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if _args.report_base:
        _base = Path(_args.report_base)
        for _name, _path in list(METHODS.items()):
            METHODS[_name] = _base / _path.relative_to(REPORT_BASE)
    for _name, _override in [("CUL", _args.cul_csv),
                             ("Class-Based", _args.class_based_csv),
                             ("Classic BL", _args.classic_bl_csv)]:
        if _override:
            METHODS[_name] = Path(_override)
    run()
