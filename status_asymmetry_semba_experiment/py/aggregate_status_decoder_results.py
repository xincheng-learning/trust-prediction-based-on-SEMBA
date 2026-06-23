from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.strict_metrics import mean_std_summary  # noqa: E402


METRICS = ["F1_negative", "PR_AUC_negative", "F1_macro", "AUROC", "F1_positive", "F1_weighted"]


def _fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _markdown_table(df: pd.DataFrame, cols) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row.get(col, "")
            cells.append(_fmt(value) if isinstance(value, (float, int)) and col not in ["seed"] else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    main = df[df["threshold_mode"] == "val_F1_macro"].copy()
    if main.empty:
        return
    main["label"] = main["model"].astype(str) + ":" + main["feature_set"].astype(str)
    for metric, name in [
        ("F1_macro", "metric_comparison.png"),
        ("F1_negative", "negative_class_comparison.png"),
    ]:
        plot_df = main.sort_values(metric, ascending=False)
        plt.figure(figsize=(9, 4.5))
        plt.bar(plot_df["label"], plot_df[metric])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric)
        plt.title(f"BitcoinOTC-1 seed=42 {metric}")
        plt.tight_layout()
        plt.savefig(figures_dir / name, dpi=160)
        plt.close()


def _write_smoke_report(df: pd.DataFrame, audit_df: pd.DataFrame, md_dir: Path, result_dir: Path) -> None:
    md_dir.mkdir(parents=True, exist_ok=True)
    main = df[df["threshold_mode"] == "val_F1_macro"].copy()
    table_cols = ["model", "feature_set", "seed", "F1_negative", "PR_AUC_negative", "F1_macro", "AUROC", "F1_positive", "F1_weighted"]
    audit_status = "pass" if not audit_df.empty and (audit_df["status"] == "pass").all() else "check"
    status_rows = main[(main["model"] == "semba-status-decoder") & (main["feature_set"] == "all_status")]
    semba_rows = main[main["model"] == "semba"]
    verdict = "未形成有效性结论"
    if not status_rows.empty and not semba_rows.empty:
        s = status_rows.iloc[0]
        b = semba_rows.iloc[0]
        deltas = {
            "F1_negative": float(s["F1_negative"] - b["F1_negative"]),
            "PR_AUC_negative": float(s["PR_AUC_negative"] - b["PR_AUC_negative"]),
            "F1_macro": float(s["F1_macro"] - b["F1_macro"]),
            "AUROC": float(s["AUROC"] - b["AUROC"]),
        }
        improved = [k for k, v in deltas.items() if v > 0]
        verdict = "初步有效，建议跑 5 seeds" if len(improved) >= 2 else "单 seed 未显示稳定优势，暂不建议直接跑 5 seeds"
    else:
        deltas = {}

    body = [
        "# BitcoinOTC-1 SEMBA-status-decoder Smoke Report",
        "",
        "## 运行范围",
        "",
        "- 数据集：BitcoinOTC-1",
        "- seed：42",
        "- 协议：strict predict-before-update",
        "- 当前阶段：smoke / minimum viable experiment",
        f"- 结果目录：`{result_dir}`",
        "",
        "## 时间安全审计",
        "",
        f"- 审计结论：`{audit_status}`",
        "- status/asymmetry 特征由严格在线历史构造；同一时间戳事件先取特征，再统一更新历史。",
        "- scaler 只在 train split 上 fit，valid/test 只 transform。",
        "",
        "## 主要指标（val_F1_macro threshold）",
        "",
        _markdown_table(main[table_cols], table_cols) if not main.empty else "无指标行。",
        "",
        "## SEMBA-status-decoder 相对 SEMBA 的差值",
        "",
        _markdown_table(pd.DataFrame([{**deltas}]), list(deltas.keys())) if deltas else "缺少 SEMBA 或 SEMBA-status-decoder 行，无法计算。",
        "",
        "## 初步判断",
        "",
        verdict,
        "",
    ]
    (md_dir / "bitcoinotc_status_decoder_smoke_report.md").write_text("\n".join(body), encoding="utf-8")


def _write_final_report(summary: pd.DataFrame, md_dir: Path) -> None:
    md_dir.mkdir(parents=True, exist_ok=True)
    cols = ["model", "feature_set", "threshold_mode"]
    metric_cols = [f"{m}_mean" for m in ["F1_negative", "PR_AUC_negative", "F1_macro", "AUROC"] if f"{m}_mean" in summary.columns]
    body = [
        "# BitcoinOTC-1 SEMBA-status-decoder Final Report",
        "",
        "当前文件由 aggregator 生成。若只有 seed=42，则该文件是 smoke 后的临时汇总，不应被解释为 5 seeds 最终结论。",
        "",
        "## 汇总指标",
        "",
        _markdown_table(summary[cols + metric_cols], cols + metric_cols) if not summary.empty else "无汇总结果。",
        "",
        "## 当前结论边界",
        "",
        "- 若 `seed_count` 小于 5，本报告只支持是否进入 5 seeds 的初步判断。",
        "- 是否支持 status/asymmetry-aware SEMBA，需要 5 seeds 和消融实验共同确认。",
        "- XGB-all 只作为参照，不作为本轮唯一成功标准。",
        "",
    ]
    (md_dir / "bitcoinotc_status_decoder_final_report.md").write_text("\n".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--result_dir", default="status_asymmetry_semba_experiment/results/BitcoinOTC-1/smoke_seed42")
    parser.add_argument("--tables_dir", default="status_asymmetry_semba_experiment/tables")
    parser.add_argument("--md_dir", default="status_asymmetry_semba_experiment/md")
    parser.add_argument("--figures_dir", default="status_asymmetry_semba_experiment/figures")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    tables_dir = Path(args.tables_dir)
    metrics_path = result_dir / "per_run_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "status_decoder_metrics_all_runs.csv", index=False)
    df[df["seed"] == 42].to_csv(tables_dir / "status_decoder_metrics_seed42.csv", index=False)
    summary = mean_std_summary(df, ["model", "feature_set", "threshold_mode"])
    summary["seed_count"] = df.groupby(["model", "feature_set", "threshold_mode"], dropna=False)["seed"].nunique().to_numpy()
    summary.to_csv(tables_dir / "status_decoder_metrics_summary.csv", index=False)
    df[df["model"] == "semba-status-decoder"].to_csv(tables_dir / "status_decoder_ablation.csv", index=False)
    audit_path = tables_dir / "status_feature_audit.csv"
    audit_df = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    _write_figures(df, Path(args.figures_dir))
    _write_smoke_report(df, audit_df, Path(args.md_dir), result_dir)
    _write_final_report(summary, Path(args.md_dir))
    print({
        "metrics": str(tables_dir / "status_decoder_metrics_seed42.csv"),
        "summary": str(tables_dir / "status_decoder_metrics_summary.csv"),
        "smoke_report": str(Path(args.md_dir) / "bitcoinotc_status_decoder_smoke_report.md"),
        "final_report": str(Path(args.md_dir) / "bitcoinotc_status_decoder_final_report.md"),
    })


if __name__ == "__main__":
    main()
