import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)): 
        return float(value)
    if isinstance(value, list):
        if not value:
            return None
        try:
            values = [float(x) for x in value]
            return sum(values) / len(values)
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def read_last_json_line(log_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(log_path):
        return None
    last_obj = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last_obj = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last_obj


def find_log_path(tag_dir: str) -> Optional[str]:
    direct = os.path.join(tag_dir, "log.txt")
    if os.path.isfile(direct):
        return direct

    candidates: List[str] = []
    for root, _, files in os.walk(tag_dir):
        if "log.txt" in files:
            candidates.append(os.path.join(root, "log.txt"))

    if not candidates:
        return None

    candidates.sort(key=lambda p: (p.count(os.sep), p))
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation test outputs and rank models")
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--tags", type=str, default="baseline,modulation_only,boundary_only,calibration_only,cucl_only,cucl_no_calibration,full_um_cucl")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    rows: List[Dict[str, Any]] = []

    for tag in tags:
        tag_dir = os.path.join(args.eval_root, tag)
        log_path = find_log_path(tag_dir)
        if log_path is None:
            rows.append(
                {
                    "tag": tag,
                    "status": "missing_log",
                    "dice_avg": None,
                    "dice_std": None,
                    "hsd_avg": None,
                    "sd_avg": None,
                    "calibration_error": None,
                    "uncertainty_error_correlation": None,
                }
            )
            continue

        stats = read_last_json_line(log_path)
        if stats is None:
            rows.append(
                {
                    "tag": tag,
                    "status": "missing_log",
                    "dice_avg": None,
                    "dice_std": None,
                    "hsd_avg": None,
                    "sd_avg": None,
                    "calibration_error": None,
                    "uncertainty_error_correlation": None,
                }
            )
            continue

        rows.append(
            {
                "tag": tag,
                "status": "ok",
                "dice_avg": to_float(stats.get("dice_avg")),
                "dice_std": to_float(stats.get("dice_std")),
                "hsd_avg": to_float(stats.get("hsd_avg")),
                "sd_avg": to_float(stats.get("sd_avg")),
                "calibration_error": to_float(stats.get("calibration_error")),
                "uncertainty_error_correlation": to_float(stats.get("uncertainty_error_correlation")),
            }
        )

    valid_rows = [r for r in rows if r["status"] == "ok" and r["dice_avg"] is not None]

    def score_key(r: Dict[str, Any]):
        return (
            -(r["dice_avg"] if r["dice_avg"] is not None else -1e9),
            r["sd_avg"] if r["sd_avg"] is not None else 1e9,
            r["hsd_avg"] if r["hsd_avg"] is not None else 1e9,
        )

    ranked = sorted(valid_rows, key=score_key)
    best_tag = ranked[0]["tag"] if ranked else None

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "status",
                "dice_avg",
                "dice_std",
                "hsd_avg",
                "sd_avg",
                "calibration_error",
                "uncertainty_error_correlation",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Ablation Test Summary\n\n")
        f.write(f"- Eval root: `{args.eval_root}`\\n")
        f.write(f"- Best model tag (ranked by dice, then sd, then hsd): **{best_tag if best_tag else 'N/A'}**\\n\n")
        f.write("| Tag | Status | Dice Avg (↑) | Dice Std | SD Avg (↓) | HD95 Avg (↓) | Calib Error (↓) | Unc-Err Corr (↑) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            dice_avg_str = "" if row["dice_avg"] is None else f"{row['dice_avg']:.4f}"
            dice_std_str = "" if row["dice_std"] is None else f"{row['dice_std']:.4f}"
            sd_avg_str = "" if row["sd_avg"] is None else f"{row['sd_avg']:.4f}"
            hsd_avg_str = "" if row["hsd_avg"] is None else f"{row['hsd_avg']:.4f}"
            calib_str = "" if row["calibration_error"] is None else f"{row['calibration_error']:.4f}"
            corr_str = "" if row["uncertainty_error_correlation"] is None else f"{row['uncertainty_error_correlation']:.4f}"
            f.write(
                f"| {row['tag']} | {row['status']} | "
                f"{dice_avg_str} | {dice_std_str} | {sd_avg_str} | {hsd_avg_str} | {calib_str} | {corr_str} |\n"
            )

        if ranked:
            f.write("\n## Ranked (Best to Worst)\n\n")
            for idx, row in enumerate(ranked, start=1):
                sd_val = "nan" if row["sd_avg"] is None else f"{row['sd_avg']:.4f}"
                hsd_val = "nan" if row["hsd_avg"] is None else f"{row['hsd_avg']:.4f}"
                f.write(
                    f"{idx}. **{row['tag']}** — Dice={row['dice_avg']:.4f}, SD={sd_val}, HD95={hsd_val}\\n"
                )

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote Markdown: {args.output_md}")
    if best_tag:
        print(f"BEST_TAG={best_tag}")


if __name__ == "__main__":
    main()
