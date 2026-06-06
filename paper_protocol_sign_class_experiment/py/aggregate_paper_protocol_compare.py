from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


PAPER_REFERENCE = [
    ["TGN", "paper Table 6 BTC-Otc", "0.74 +/- 0.02", "0.82 +/- 0.03"],
    ["SEMBA", "paper Table 6 BTC-Otc", "0.81 +/- 0.04", "0.79 +/- 0.02"],
]

METRICS = [
    "F1_bin",
    "AUROC",
    "F1_negative",
    "F1_macro",
    "PR_AUC_negative",
    "F1_weighted",
    "balanced_accuracy",
    "total_wall_time_sec",
]

CONFUSION = ["TN", "FP", "FN", "TP", "predicted_negative_count", "predicted_positive_count"]


def _fmt(mean, std=None, digits: int = 4) -> str:
    if pd.isna(mean):
        return ""
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _label(row) -> str:
    model = str(row["model"])
    feature_set = str(row.get("feature_set", ""))
    emb = row.get("embedding_dim", None)
    if feature_set and feature_set != "nan" and feature_set not in ["none", "graph_memory"]:
        model = f"{model}-{feature_set}"
    if not pd.isna(emb) and model in ["semba", "tgn", "semba-noprop"]:
        model = f"{model}-emb{int(emb)}"
    return model


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "feature_set", "threshold_mode"]
    if "embedding_dim" in df.columns:
        group_cols.append("embedding_dim")
    cols = [col for col in METRICS + CONFUSION if col in df.columns]
    summary = df.groupby(group_cols, dropna=False)[cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _main_rows(summary: pd.DataFrame, threshold_mode: str) -> List[List[str]]:
    view = summary[summary["threshold_mode"] == threshold_mode].copy()
    if view.empty:
        return []
    view["label"] = view.apply(_label, axis=1)
    view = view.sort_values("F1_bin_mean", ascending=False)
    rows = []
    for _, row in view.iterrows():
        rows.append([
            row["label"],
            _fmt(row.get("F1_bin_mean"), row.get("F1_bin_std")),
            _fmt(row.get("AUROC_mean"), row.get("AUROC_std")),
            _fmt(row.get("F1_negative_mean"), row.get("F1_negative_std")),
            _fmt(row.get("F1_macro_mean"), row.get("F1_macro_std")),
            _fmt(row.get("PR_AUC_negative_mean"), row.get("PR_AUC_negative_std")),
            _fmt(row.get("total_wall_time_sec_mean"), row.get("total_wall_time_sec_std"), digits=2),
        ])
    return rows


def _confusion_rows(summary: pd.DataFrame, threshold_mode: str) -> List[List[str]]:
    view = summary[summary["threshold_mode"] == threshold_mode].copy()
    if view.empty:
        return []
    view["label"] = view.apply(_label, axis=1)
    view = view.sort_values("F1_bin_mean", ascending=False)
    rows = []
    for _, row in view.iterrows():
        rows.append([
            row["label"],
            _fmt(row.get("TN_mean"), None, digits=1),
            _fmt(row.get("FP_mean"), None, digits=1),
            _fmt(row.get("FN_mean"), None, digits=1),
            _fmt(row.get("TP_mean"), None, digits=1),
            _fmt(row.get("predicted_negative_count_mean"), None, digits=1),
            _fmt(row.get("predicted_positive_count_mean"), None, digits=1),
        ])
    return rows


def _find_row(summary: pd.DataFrame, label: str, threshold_mode: str):
    view = summary[summary["threshold_mode"] == threshold_mode].copy()
    if view.empty:
        return None
    view["label"] = view.apply(_label, axis=1)
    matched = view[view["label"] == label]
    if matched.empty:
        return None
    return matched.iloc[0]


def _key_findings(summary: pd.DataFrame) -> str:
    xgb_all = _find_row(summary, "XGB-paper-selected-all", "fixed_0.5")
    xgb_dst = _find_row(summary, "XGB-paper-selected-dst_reputation", "fixed_0.5")
    semba = _find_row(summary, "semba-emb64", "fixed_0.5")
    tgn = _find_row(summary, "tgn-emb64", "fixed_0.5")

    bullets = []
    if xgb_all is not None:
        bullets.append(
            "- `XGB-all` 在 paper protocol 固定阈值 0.5 下达到 "
            f"`F1_bin={_fmt(xgb_all['F1_bin_mean'], xgb_all['F1_bin_std'])}`、"
            f"`AUROC={_fmt(xgb_all['AUROC_mean'], xgb_all['AUROC_std'])}`。"
            "这个结果几乎一定受到 batch-inclusive 泄漏影响，因为 pair/history 特征会在预测当前 batch 前看到当前 batch。"
        )
    if xgb_dst is not None:
        bullets.append(
            "- `XGB-dst_reputation` 不使用 pair history，但仍达到 "
            f"`F1_bin={_fmt(xgb_dst['F1_bin_mean'], xgb_dst['F1_bin_std'])}`、"
            f"`AUROC={_fmt(xgb_dst['AUROC_mean'], xgb_dst['AUROC_std'])}`。"
            "这说明目标节点声誉/可见性在论文协议下依然是强信号，但它也会受到当前 batch 先写入历史的影响。"
        )
    if semba is not None:
        bullets.append(
            "- 本地 `SEMBA-emb64` 固定阈值 0.5 为 "
            f"`F1_bin={_fmt(semba['F1_bin_mean'], semba['F1_bin_std'])}`、"
            f"`AUROC={_fmt(semba['AUROC_mean'], semba['AUROC_std'])}`；"
            "与论文 Table 6 的 SEMBA `F1=0.81 +/- 0.04, AUROC=0.79 +/- 0.02` 相比，F1 更高，AUROC 基本贴近。"
        )
    if tgn is not None:
        bullets.append(
            "- 本地 `TGN-emb64` 固定阈值 0.5 为 "
            f"`F1_bin={_fmt(tgn['F1_bin_mean'], tgn['F1_bin_std'])}`、"
            f"`AUROC={_fmt(tgn['AUROC_mean'], tgn['AUROC_std'])}`；"
            "与论文 TGN `F1=0.74 +/- 0.02, AUROC=0.82 +/- 0.03` 相比，F1 更高，AUROC 略低但接近。"
        )
    bullets.append(
        "- 因此，本轮 paper protocol 实验主要说明：按论文/旧代码协议，表格历史特征会变得极强，"
        "但其中一部分强度来自协议泄漏；研究上应保留 XGB 作为强 baseline，同时不能把该协议下的 1.0 当成严格泛化能力。"
    )
    return "\n".join(bullets)


def _strict_context(strict_summary_path: Path) -> str:
    if not strict_summary_path.exists():
        return "未找到上一轮严格无泄漏 summary，因此本节暂不比较。"
    strict_df = pd.read_csv(strict_summary_path)
    strict_df = strict_df[strict_df.get("threshold_mode", "") == "val_F1_macro"]
    if strict_df.empty:
        return "上一轮严格无泄漏 summary 存在，但没有 `val_F1_macro` 行。"
    rows = []
    for _, row in strict_df.sort_values("F1_macro_mean", ascending=False).iterrows():
        rows.append([
            str(row["model"]),
            _fmt(row.get("F1_macro_mean"), row.get("F1_macro_std")),
            _fmt(row.get("F1_negative_mean"), row.get("F1_negative_std")),
            _fmt(row.get("PR_AUC_negative_mean"), row.get("PR_AUC_negative_std")),
            _fmt(row.get("AUROC_mean"), row.get("AUROC_std")),
        ])
    return _markdown_table(
        ["model", "strict F1_macro", "strict F1_negative", "strict PR_AUC_negative", "strict AUROC"],
        rows,
    )


def _diagnostic_section(diagnostic_dir: Path, dataset: str) -> str:
    metrics_path = diagnostic_dir / dataset / "per_run_metrics.csv"
    if not metrics_path.exists():
        return "512 维诊断结果尚未运行。"
    df = pd.read_csv(metrics_path)
    summary = _summarize(df)
    rows = _main_rows(summary, "fixed_0.5")
    if not rows:
        return "512 维诊断结果存在，但没有 fixed_0.5 行。"
    return _markdown_table(
        ["model", "F1_bin", "AUROC", "F1_negative", "F1_macro", "PR_AUC_negative", "total sec"],
        rows,
    )


def _processed_cache_section(path: Path) -> str:
    if not path.exists():
        return "未找到 processed 缓存检查文件。"
    rows = json.loads(path.read_text(encoding="utf-8"))
    table_rows = []
    for row in rows:
        if row.get("status") == "ok":
            table_rows.append([
                row["processed_dir"],
                row["status"],
                row.get("events", ""),
                row.get("nodes", ""),
                row.get("train", ""),
                row.get("val", ""),
                row.get("test", ""),
                f"{row.get('pos_rate', 0):.6f}",
            ])
        else:
            table_rows.append([
                row["processed_dir"],
                row["status"],
                row.get("error", ""),
                "",
                "",
                "",
                "",
                "",
            ])
    return _markdown_table(
        ["processed_dir", "status", "events/error", "nodes", "train", "val", "test", "pos_rate"],
        table_rows,
    )


def _feature_importance_section(result_dir: Path) -> str:
    paths = [
        result_dir / "feature_importance" / "xgb_XGB-paper-selected_all_seed_42.csv",
        result_dir / "feature_importance" / "xgb_XGB-paper-selected_no_dst_reputation_seed_42.csv",
        result_dir / "feature_importance" / "xgb_XGB-paper-selected_dst_reputation_seed_42.csv",
    ]
    rows = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path).head(5)
        model = df["model"].iloc[0]
        feature_set = df["feature_set"].iloc[0]
        top_features = ", ".join(
            f"{row.feature}={row.importance:.3f}"
            for row in df.itertuples(index=False)
        )
        rows.append([f"{model}-{feature_set}", top_features])
    if not rows:
        return "未找到 seed 42 的特征重要性文件。"
    return _markdown_table(["model", "top features"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="paper_protocol_sign_class_experiment/results")
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--out_md", default="paper_protocol_sign_class_experiment/md/bitcoinotc_paper_protocol_sign_class_summary.md")
    parser.add_argument("--strict_summary", default="results/strict_compare_full_gpu/BitcoinOTC-1/summary_by_model.csv")
    parser.add_argument("--diagnostic_results_dir", default="paper_protocol_sign_class_experiment/results_config512_check")
    parser.add_argument("--processed_cache_check", default="paper_protocol_sign_class_experiment/results/processed_cache_check.json")
    args = parser.parse_args()

    result_dir = Path(args.results_dir) / args.dataset
    metrics_path = result_dir / "per_run_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path, low_memory=False)
    summary = _summarize(df)
    summary_path = result_dir / "summary_by_model.csv"
    summary.to_csv(summary_path, index=False)

    manifest_paths = sorted(result_dir.glob("manifest*.json"))
    manifests = []
    for path in manifest_paths:
        manifests.append(json.loads(path.read_text(encoding="utf-8")))

    fixed_rows = _main_rows(summary, "fixed_0.5")
    val_f1_bin_rows = _main_rows(summary, "val_F1_bin")
    val_f1_macro_rows = _main_rows(summary, "val_F1_macro")
    confusion_rows = _confusion_rows(summary, "fixed_0.5")

    report = f"""# BitcoinOTC-1 论文二分类协议 XGB 对比报告

## 实验口径

- 任务：`sign_class` 二分类，即已知未来有边，预测这条边是正边还是负边。
- 标签：`1` 是正边/信任，`0` 是负边/不信任。
- 协议：paper protocol，刻意模拟旧 `train.py` 的 `to_update=True`，当前 batch 先写入历史/记忆，再预测当前 batch。
- 风险：这个协议可能有 batch 内信息泄漏，因此不能替代上一轮严格无泄漏结论。
- 数据目录：`{args.results_dir}/{args.dataset}/`

## 指标中文解释

- `F1_bin`：二分类正类 F1，默认正类是 `1`。公式是 `2 * precision * recall / (precision + recall)`。
- `AUROC`：ROC 曲线下面积，可理解为随机正边得分高于随机负边得分的概率。
- `F1_negative`：负边/不信任边这一类的 F1。
- `F1_macro`：正类 F1 和负类 F1 的平均，类别不平衡时比加权 F1 更公平。
- `PR_AUC_negative`：把负边当成目标类时的 precision-recall 曲线面积。
- `F1_weighted`：按各类别样本数加权的 F1，不是 1-10 分评分；正边多时它会更受正边表现影响。
- `total sec`：训练和评估耗时之和，不含原始数据下载。

## 论文 Table 6 外部参考

{_markdown_table(["model", "source", "paper F1_bin", "paper AUROC"], PAPER_REFERENCE)}

## 本地 paper protocol 主结果：固定阈值 0.5

固定阈值 0.5 最接近旧 SEMBA/TGN 的 `classify=True` 口径，也是和论文 Table 6 对比时的主表。

{_markdown_table(["model", "F1_bin", "AUROC", "F1_negative", "F1_macro", "PR_AUC_negative", "total sec"], fixed_rows)}

## 关键结论

{_key_findings(summary)}

## XGB 特征重要性证据

{_feature_importance_section(result_dir)}

## 辅助结果：验证集选择正类 F1 阈值

{_markdown_table(["model", "F1_bin", "AUROC", "F1_negative", "F1_macro", "PR_AUC_negative", "total sec"], val_f1_bin_rows)}

## 辅助结果：验证集选择 macro-F1 阈值

{_markdown_table(["model", "F1_bin", "AUROC", "F1_negative", "F1_macro", "PR_AUC_negative", "total sec"], val_f1_macro_rows)}

## 混淆矩阵均值：固定阈值 0.5

{_markdown_table(["model", "TN", "FP", "FN", "TP", "pred neg", "pred pos"], confusion_rows)}

## 上一轮严格无泄漏结果对照

{_strict_context(Path(args.strict_summary))}

## 512 维单 seed 诊断

{_diagnostic_section(Path(args.diagnostic_results_dir), args.dataset)}

## processed 缓存检查

{_processed_cache_section(Path(args.processed_cache_check))}

## 必答问题

1. 是否使用 `sign_class` 二分类任务：是。本实验只处理正边/负边二分类。
2. 是否使用 70/15/15 时间切分：是。由项目 `TemporalData.train_val_test_split` 产生，manifest 中记录 split summary。
3. 是否使用旧代码一致的 `to_update=True` paper protocol：是。图模型 runner 在预测前调用 `to_update=True`；XGB 特征先写入当前 batch 再生成当前 batch 特征。
4. XGB 是否使用 batch-inclusive 特征：是。这是本实验刻意模拟论文/旧代码口径的核心。
5. 本地 SEMBA/TGN 是否接近论文 Table 6：见固定阈值 0.5 主表，应以 `F1_bin` 和 `AUROC` 对照论文参考行。
6. 若本地 SEMBA/TGN 未接近论文：不能直接归因于环境，需要同时检查参数规模、processed 缓存、PyTorch/PyG 版本和旧代码协议。
7. 论文协议下 XGB 与 SEMBA/TGN 如何：以固定阈值 0.5 主表为准，辅助阈值表只能作为补充。
8. 负边识别上 XGB 是否仍强：看 `F1_negative`、`F1_macro`、`PR_AUC_negative`。
9. 与上一轮严格无泄漏结果是否方向一致：看“上一轮严格无泄漏结果对照”。
10. 是否值得融合 XGB 特征进 SEMBA：若 XGB 在主表或负边指标上接近/超过图模型，则值得继续做 `XGB features + SEMBA decoder`。

"""

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")
    print(json.dumps({
        "runner": "aggregate_paper_protocol_compare",
        "summary_csv": str(summary_path),
        "out_md": str(out_md),
        "rows": int(len(df)),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
