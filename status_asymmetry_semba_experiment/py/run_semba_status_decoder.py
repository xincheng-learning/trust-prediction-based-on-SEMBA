from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.strict_metrics import choose_threshold, classification_metrics, mean_std_summary  # noqa: E402
from experiments.strict_temporal_protocol import iter_time_blocks, split_summary  # noqa: E402
from model_wrapper import STGNN  # noqa: E402
from status_asymmetry_semba_experiment.py.build_online_status_features import (  # noqa: E402
    FEATURE_GROUPS,
    build_status_feature_frames,
    status_feature_audit_rows,
    write_feature_dictionary,
)
from utils import get_data  # noqa: E402


class StatusPairDecoder(torch.nn.Module):
    def __init__(self, embedding_dim: int, status_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4 * embedding_dim + status_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, status_features: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([z_src, z_dst, torch.abs(z_src - z_dst), z_src * z_dst, status_features], dim=1)
        return self.net(pair).squeeze(-1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_node_features(data, feat_type: str, num_feats: int, device: str):
    if feat_type == "zeros":
        return torch.zeros(data.num_nodes, num_feats, dtype=torch.float, device=device), num_feats
    if feat_type == "random":
        return torch.rand(data.num_nodes, num_feats, dtype=torch.float, device=device), num_feats
    if feat_type == "one-hot":
        return torch.diag(torch.ones(data.num_nodes, dtype=torch.float, device=device)), data.num_nodes
    raise ValueError(f"Unsupported feat_type: {feat_type}")


def _edge_parts(batch, device: str) -> Dict[str, torch.Tensor]:
    src = batch.src.to(device)
    dst = batch.dst.to(device)
    t = batch.t.to(device)
    weight = batch.msg.to(device)
    signs = batch.y.to(device).long()
    pos_mask = signs == 1
    neg_mask = signs == 0
    return {
        "src": src,
        "dst": dst,
        "t": t,
        "weight": weight,
        "signs": signs,
        "pos_edge_index": torch.stack([src[pos_mask], dst[pos_mask]]),
        "neg_edge_index": torch.stack([src[neg_mask], dst[neg_mask]]),
        "pos_times": t[pos_mask],
        "neg_times": t[neg_mask],
        "pos_weights": weight[pos_mask],
        "neg_weights": weight[neg_mask],
    }


def _time_block_offsets(data, max_events_per_block: Optional[int]) -> Iterator[Tuple[int, int, object]]:
    start = 0
    for batch in iter_time_blocks(data, max_events_per_block):
        end = start + int(batch.num_events)
        yield start, end, batch
        start = end


def _predict_embedding_before_update(model: STGNN, x: torch.Tensor, batch, device: str):
    parts = _edge_parts(batch, device)
    z = model(
        x,
        parts["pos_edge_index"],
        parts["neg_edge_index"],
        parts["pos_times"],
        parts["neg_times"],
        parts["pos_weights"],
        parts["neg_weights"],
        to_update=False,
    )
    return parts, z


def _update_after_prediction(model: STGNN, parts: Dict[str, torch.Tensor]) -> None:
    model.update_memory(
        parts["pos_edge_index"],
        parts["neg_edge_index"],
        parts["pos_times"],
        parts["neg_times"],
        parts["pos_weights"],
        parts["neg_weights"],
    )
    if hasattr(model, "mem_model") and hasattr(model.mem_model, "detach"):
        model.mem_model.detach()
    if hasattr(model, "model") and hasattr(model.model, "detach"):
        model.model.detach()


def _status_tensor(status_df: pd.DataFrame, start: int, end: int, device: str) -> torch.Tensor:
    return torch.tensor(status_df.iloc[start:end].to_numpy(dtype=np.float32), dtype=torch.float, device=device)


def _probabilities(model_kind: str, model: STGNN, decoder, z, parts, status_features):
    if model_kind == "semba-status-decoder":
        logits = decoder(z[parts["src"]], z[parts["dst"]], status_features)
        return torch.sigmoid(logits), logits
    prob = model.predict(z, parts["src"], parts["dst"])
    return torch.clamp(prob, 1e-6, 1 - 1e-6), None


def _train_epoch(model_kind: str, model: STGNN, decoder, x, train_data, status_df, optimizer, args):
    model.train()
    if decoder is not None:
        decoder.train()
    total_loss = 0.0
    total_examples = 0
    all_y: List[int] = []
    all_prob: List[float] = []

    for start, end, batch in _time_block_offsets(train_data, args.max_events_per_block):
        optimizer.zero_grad()
        parts, z = _predict_embedding_before_update(model, x, batch, args.device)
        status_features = _status_tensor(status_df, start, end, args.device)
        target = parts["signs"].float()
        prob, logits = _probabilities(model_kind, model, decoder, z, parts, status_features)
        sample_weight = torch.where(
            target == 0,
            torch.full_like(target, args.neg_wt),
            torch.ones_like(target),
        )
        if logits is not None:
            loss = F.binary_cross_entropy_with_logits(logits, target, weight=sample_weight)
        else:
            loss = F.binary_cross_entropy(prob, target, weight=sample_weight)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _update_after_prediction(model, parts)

        n = int(target.numel())
        total_loss += float(loss.detach()) * n
        total_examples += n
        all_y.extend(target.detach().cpu().numpy().astype(int).tolist())
        all_prob.extend(prob.detach().cpu().numpy().tolist())
    return total_loss / max(1, total_examples), all_y, all_prob


@torch.no_grad()
def _replay_observed(model: STGNN, data, args) -> None:
    for batch in iter_time_blocks(data, args.max_events_per_block):
        parts = _edge_parts(batch, args.device)
        _update_after_prediction(model, parts)


@torch.no_grad()
def _evaluate_split(model_kind: str, model: STGNN, decoder, x, histories, eval_data, status_df, args):
    model.eval()
    if decoder is not None:
        decoder.eval()
    model.reset_memory()
    for history in histories:
        _replay_observed(model, history, args)

    all_y: List[int] = []
    all_prob: List[float] = []
    started = time.time()
    for start, end, batch in _time_block_offsets(eval_data, args.max_events_per_block):
        parts, z = _predict_embedding_before_update(model, x, batch, args.device)
        status_features = _status_tensor(status_df, start, end, args.device)
        prob, _ = _probabilities(model_kind, model, decoder, z, parts, status_features)
        all_y.extend(parts["signs"].detach().cpu().numpy().astype(int).tolist())
        all_prob.extend(prob.detach().cpu().numpy().tolist())
        _update_after_prediction(model, parts)
    return all_y, all_prob, time.time() - started


def _model_param_count(model: torch.nn.Module, decoder=None) -> int:
    total = int(sum(p.numel() for p in model.parameters()))
    if decoder is not None:
        total += int(sum(p.numel() for p in decoder.parameters()))
    return total


def _append_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path, low_memory=False)
        pd.concat([existing, df], ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def _make_model(model_kind: str, data, num_feats: int, status_dim: int, args):
    base_name = "semba" if model_kind == "semba-status-decoder" else model_kind
    model = STGNN(
        base_name,
        "sign_class",
        num_feats,
        data.num_nodes,
        args.embedding_dim,
        args.num_layers,
        device=args.device,
        debug=False,
    ).to(args.device)
    decoder = None
    if model_kind == "semba-status-decoder":
        decoder = StatusPairDecoder(args.embedding_dim, status_dim, args.status_hidden_dim, args.status_dropout).to(args.device)
    return model, decoder


def _run_one(model_kind: str, feature_set: str, seed: int, data, train_data, val_data, test_data, status_frames, args, out_dir: Path):
    _set_seed(seed)
    x, num_feats = _make_node_features(data, args.feat_type, args.num_feats, args.device)
    status_dim = len(status_frames["feature_columns"]) if model_kind == "semba-status-decoder" else 0
    model, decoder = _make_model(model_kind, data, num_feats, status_dim, args)
    params = list(model.parameters()) + (list(decoder.parameters()) if decoder is not None else [])
    optimizer = torch.optim.Adam(params, lr=args.lr_init)

    best_state = None
    best_decoder_state = None
    best_epoch = 0
    best_val_score = -1.0
    stale_epochs = 0
    train_started = time.time()
    train_history: List[Dict[str, object]] = []

    for epoch in range(1, args.num_epochs + 1):
        model.reset_memory()
        loss, _, _ = _train_epoch(
            model_kind,
            model,
            decoder,
            x,
            train_data,
            status_frames["train"],
            optimizer,
            args,
        )
        val_y, val_prob, val_eval_sec = _evaluate_split(
            model_kind,
            model,
            decoder,
            x,
            [train_data],
            val_data,
            status_frames["val"],
            args,
        )
        val_threshold, val_score = choose_threshold(val_y, val_prob, args.threshold_metric)
        val_metrics = classification_metrics(val_y, val_prob, val_threshold)
        train_history.append({
            "epoch": epoch,
            "loss": loss,
            "val_threshold": val_threshold,
            "val_threshold_score": val_score,
            "val_F1_negative": val_metrics["F1_negative"],
            "val_PR_AUC_negative": val_metrics["PR_AUC_negative"],
            "val_F1_macro": val_metrics["F1_macro"],
            "val_AUROC": val_metrics["AUROC"],
            "val_eval_sec": val_eval_sec,
        })
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_decoder_state = copy.deepcopy(decoder.state_dict()) if decoder is not None else None
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stop_patience:
            break

    train_sec = time.time() - train_started
    if best_state is not None:
        model.load_state_dict(best_state)
    if decoder is not None and best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)

    val_y, val_prob, _ = _evaluate_split(model_kind, model, decoder, x, [train_data], val_data, status_frames["val"], args)
    threshold, threshold_score = choose_threshold(val_y, val_prob, args.threshold_metric)
    test_y, test_prob, eval_sec = _evaluate_split(
        model_kind,
        model,
        decoder,
        x,
        [train_data, val_data],
        test_data,
        status_frames["test"],
        args,
    )

    metric_rows = []
    for threshold_mode, thr in [("val_F1_macro", threshold), ("fixed_0.5", 0.5)]:
        metrics = classification_metrics(test_y, test_prob, threshold=thr)
        metric_rows.append({
            **metrics,
            "dataset": args.dataset,
            "model": model_kind,
            "feature_set": feature_set if model_kind == "semba-status-decoder" else "graph_memory",
            "seed": seed,
            "threshold_mode": threshold_mode,
            "best_epoch": best_epoch,
            "val_threshold_score": threshold_score,
            "train_wall_time_sec": round(train_sec, 6),
            "eval_wall_time_sec": round(eval_sec, 6),
            "total_wall_time_sec": round(train_sec + eval_sec, 6),
            "device": args.device,
            "num_parameters": _model_param_count(model, decoder),
            "max_events_per_block": args.max_events_per_block,
            "status_feature_count": status_dim,
        })

    prediction_rows = [{
        "dataset": args.dataset,
        "model": model_kind,
        "feature_set": feature_set if model_kind == "semba-status-decoder" else "graph_memory",
        "seed": seed,
        "row_id": idx,
        "y_true": int(y),
        "y_prob": float(prob),
        "threshold": float(threshold),
        "y_pred": int(prob >= threshold),
    } for idx, (y, prob) in enumerate(zip(test_y, test_prob))]

    pd.DataFrame(train_history).to_csv(out_dir / "training_curves" / f"{model_kind}_{feature_set}_seed_{seed}.csv", index=False)
    _append_csv(out_dir / "per_run_metrics.csv", metric_rows)
    _append_csv(out_dir / "per_run_predictions.csv", prediction_rows)
    return metric_rows


def _copy_xgb_reference(args, out_dir: Path) -> List[Dict[str, object]]:
    ref_path = ROOT / "results" / "strict_compare_full_gpu" / args.dataset / "per_run_metrics.csv"
    if not ref_path.exists():
        return []
    df = pd.read_csv(ref_path)
    mask = (
        (df["model"] == "XGB-all")
        & (df["feature_set"] == "all")
        & (df["seed"].isin(args.seeds))
        & (df["threshold_mode"].isin(["val_F1_macro", "fixed_0.5"]))
    )
    rows = df.loc[mask].copy()
    if rows.empty:
        return []
    rows["model"] = "XGB-all-reference"
    rows["note"] = "copied from existing strict_compare_full_gpu; reference only"
    _append_csv(out_dir / "per_run_metrics.csv", rows.to_dict("records"))
    return rows.to_dict("records")


def _write_manifest(out_dir: Path, args, split_rows, feature_columns):
    manifest = {
        "runner": "run_semba_status_decoder",
        "dataset": args.dataset,
        "processed_dir": args.processed_dir,
        "max_events": args.max_events,
        "smoke": args.smoke,
        "models": args.models,
        "feature_sets": args.feature_sets,
        "seeds": args.seeds,
        "num_epochs": args.num_epochs,
        "max_events_per_block": args.max_events_per_block,
        "protocol": "strict predict-before-update; status features built from history before current timestamp/event",
        "feature_columns": feature_columns,
        "split_summary": split_rows,
    }
    (out_dir / "manifest_status_decoder.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_tables(result_dir: Path, tables_dir: Path):
    metrics_path = result_dir / "per_run_metrics.csv"
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "status_decoder_metrics_seed42.csv", index=False)
    mean_std_summary(df[df["threshold_mode"] == "val_F1_macro"], ["model", "feature_set", "threshold_mode"]).to_csv(
        tables_dir / "status_decoder_metrics_summary.csv",
        index=False,
    )
    df[df["model"] == "semba-status-decoder"].to_csv(tables_dir / "status_decoder_ablation.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "--data", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results_dir", default="status_asymmetry_semba_experiment/results")
    parser.add_argument("--models", nargs="+", choices=["semba", "semba-noprop", "tgn", "semba-status-decoder"], default=["semba", "semba-status-decoder"])
    parser.add_argument("--feature_sets", nargs="+", choices=sorted(FEATURE_GROUPS), default=["all_status"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--max_events_per_block", type=int, default=1000)
    parser.add_argument("--threshold_metric", default="F1_macro")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feat_type", default="zeros")
    parser.add_argument("--num_feats", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr_init", type=float, default=0.01)
    parser.add_argument("--neg_wt", type=float, default=1.0)
    parser.add_argument("--status_hidden_dim", type=int, default=128)
    parser.add_argument("--status_dropout", type=float, default=0.1)
    parser.add_argument("--include_xgb_reference", action="store_true", default=True)
    args = parser.parse_args()

    if args.smoke:
        if args.max_events is None:
            args.max_events = 12000
        args.num_epochs = min(args.num_epochs, 3)
        args.early_stop_patience = min(args.early_stop_patience, 2)

    data, train_data, val_data, test_data = get_data(
        args.dataset,
        f"./data/{args.dataset}",
        args.device,
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )

    out_dir = Path(args.results_dir) / args.dataset / ("smoke_seed42" if args.smoke else "full")
    (out_dir / "training_curves").mkdir(parents=True, exist_ok=True)
    tables_dir = ROOT / "status_asymmetry_semba_experiment" / "tables"
    write_feature_dictionary(tables_dir / "status_feature_dictionary.csv")

    split_rows = split_summary(data, train_data, val_data, test_data)
    rows: List[Dict[str, object]] = []
    first_feature_columns: List[str] = []
    for feature_set in args.feature_sets:
        status_frames = build_status_feature_frames(data, train_data, val_data, feature_set=feature_set)
        first_feature_columns = first_feature_columns or list(status_frames["feature_columns"])
        audit_rows = status_feature_audit_rows(status_frames, train_data, val_data, test_data)
        pd.DataFrame(audit_rows).to_csv(tables_dir / "status_feature_audit.csv", index=False)
        for model_kind in args.models:
            if model_kind != "semba-status-decoder" and feature_set != args.feature_sets[0]:
                continue
            effective_feature_set = feature_set if model_kind == "semba-status-decoder" else "none"
            for seed in args.seeds:
                model_rows = _run_one(model_kind, effective_feature_set, seed, data, train_data, val_data, test_data, status_frames, args, out_dir)
                rows.extend(model_rows)
                print(json.dumps({"model": model_kind, "feature_set": effective_feature_set, "seed": seed, "rows": model_rows}, indent=2, ensure_ascii=False))

    if args.include_xgb_reference and not args.smoke:
        rows.extend(_copy_xgb_reference(args, out_dir))

    _write_manifest(out_dir, args, split_rows, first_feature_columns)
    _write_tables(out_dir, tables_dir)
    print(json.dumps({"runner": "run_semba_status_decoder", "out_dir": str(out_dir), "rows_written": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
