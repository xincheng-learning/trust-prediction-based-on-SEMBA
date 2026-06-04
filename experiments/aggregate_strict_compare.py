from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.strict_metrics import mean_std_summary


PRIMARY_METRICS = [
    "PR_AUC_negative",
    "F1_negative",
    "F1_macro",
    "balanced_accuracy",
    "AUROC",
    "F1_weighted",
    "train_wall_time_sec",
    "eval_wall_time_sec",
    "total_wall_time_sec",
]


def _format_mean_std(row, metric):
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if pd.isna(std):
        std = 0.0
    return f"{mean:.4f} +/- {std:.4f}" if metric not in {
        "train_wall_time_sec",
        "eval_wall_time_sec",
        "total_wall_time_sec",
    } else f"{mean:.2f} +/- {std:.2f}"


def _markdown_table(summary):
    headers = ["model", "threshold", "seeds"] + PRIMARY_METRICS
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in summary.iterrows():
        values = [
            str(row["model"]),
            str(row["threshold_mode"]),
            str(int(row["seed_count"])),
        ]
        values.extend(_format_mean_std(row, metric) for metric in PRIMARY_METRICS)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/strict_compare_pilot")
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument(
        "--out_md",
        default="strict_xgb_comparison_experiment/bitcoinotc_strict_compare_summary.md",
    )
    parser.add_argument("--threshold_mode", default="val_F1_macro")
    args = parser.parse_args()

    dataset_dir = Path(args.results_dir) / args.dataset
    metrics_path = dataset_dir / "per_run_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path)
    filtered = df[df["threshold_mode"] == args.threshold_mode].copy()
    summary = mean_std_summary(filtered, ["model", "threshold_mode"])
    seed_counts = filtered.groupby(["model", "threshold_mode"], dropna=False)["seed"].nunique().reset_index(name="seed_count")
    summary = summary.merge(seed_counts, on=["model", "threshold_mode"], how="left")
    summary = summary.sort_values(["PR_AUC_negative_mean", "F1_negative_mean"], ascending=False)
    summary.to_csv(dataset_dir / "summary_by_model.csv", index=False)

    md = [
        "# BitcoinOTC Strict XGB Comparison Summary",
        "",
        f"Results directory: `{dataset_dir}`",
        f"Threshold mode: `{args.threshold_mode}`",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Notes",
        "",
        "- Main ranking uses `PR_AUC_negative` first, then negative-class F1.",
        "- `F1_weighted` is reported only as a secondary class-imbalance-sensitive metric.",
        "- Graph rows use predict-before-update no-leakage mini-blocks.",
    ]
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {dataset_dir / 'summary_by_model.csv'}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
