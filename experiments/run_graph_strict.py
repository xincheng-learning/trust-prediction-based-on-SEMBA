from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.strict_metrics import choose_threshold, classification_metrics
from experiments.strict_temporal_protocol import iter_time_blocks, split_summary
from model_wrapper import STGNN
from utils import get_data


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_node_features(data, feat_type: str, num_feats: int, device: str):
    if feat_type == "zeros":
        return torch.zeros(data.num_nodes, num_feats, dtype=torch.float, device=device), num_feats
    if feat_type == "random":
        return torch.rand(data.num_nodes, num_feats, dtype=torch.float, device=device), num_feats
    if feat_type == "one-hot":
        return torch.diag(torch.ones(data.num_nodes, dtype=torch.float, device=device)), data.num_nodes
    raise ValueError(f"Unsupported feat_type: {feat_type}")


def _edge_parts(batch, device: str):
    src = batch.src.to(device)
    dst = batch.dst.to(device)
    t = batch.t.to(device)
    weight = batch.msg.to(device)
    signs = batch.y.to(device).long()

    pos_mask = signs == 1
    neg_mask = signs == 0
    pos_edge_index = torch.stack([src[pos_mask], dst[pos_mask]])
    neg_edge_index = torch.stack([src[neg_mask], dst[neg_mask]])
    return {
        "src": src,
        "dst": dst,
        "t": t,
        "weight": weight,
        "signs": signs,
        "pos_edge_index": pos_edge_index,
        "neg_edge_index": neg_edge_index,
        "pos_times": t[pos_mask],
        "neg_times": t[neg_mask],
        "pos_weights": weight[pos_mask],
        "neg_weights": weight[neg_mask],
    }


def _predict_before_update(model, x, batch, device: str):
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
    prob = model.predict(z, parts["src"], parts["dst"])
    return parts, prob


def _update_after_prediction(model, parts):
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


def _train_epoch(model, x, train_data, args):
    model.train()
    optimizer = args.optimizer
    total_loss = 0.0
    total_examples = 0
    all_y = []
    all_prob = []

    for batch in iter_time_blocks(train_data, args.max_events_per_block):
        optimizer.zero_grad()
        parts, prob = _predict_before_update(model, x, batch, args.device)
        target = parts["signs"].float()
        prob = torch.clamp(prob, 1e-6, 1 - 1e-6)
        sample_weight = torch.where(
            target == 0,
            torch.full_like(target, args.neg_wt),
            torch.ones_like(target),
        )
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
def _replay_observed(model, x, data, args):
    for batch in iter_time_blocks(data, args.max_events_per_block):
        parts = _edge_parts(batch, args.device)
        _update_after_prediction(model, parts)


@torch.no_grad()
def _evaluate_split(model, x, history_splits, eval_data, args):
    model.eval()
    model.reset_memory()
    for history in history_splits:
        _replay_observed(model, x, history, args)

    all_y = []
    all_prob = []
    started = time.time()
    for batch in iter_time_blocks(eval_data, args.max_events_per_block):
        parts, prob = _predict_before_update(model, x, batch, args.device)
        target = parts["signs"].detach().cpu().numpy().astype(int).tolist()
        all_y.extend(target)
        all_prob.extend(prob.detach().cpu().numpy().tolist())
        _update_after_prediction(model, parts)
    return all_y, all_prob, time.time() - started


def _model_param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def _append_csv(path: Path, rows):
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        pd.concat([existing, df], ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def _run_one(model_name: str, seed: int, data, train_data, val_data, test_data, args, out_dir: Path):
    _set_seed(seed)
    x, num_feats = _make_node_features(data, args.feat_type, args.num_feats, args.device)
    model = STGNN(
        model_name,
        "sign_class",
        num_feats,
        data.num_nodes,
        args.embedding_dim,
        args.num_layers,
        device=args.device,
        debug=False,
    )
    model.to(args.device)
    optimizer = torch.optim.Adam(set(model.parameters()), lr=args.lr_init)
    args.optimizer = optimizer

    best_state = None
    best_epoch = 0
    best_val_score = -1.0
    stale_epochs = 0
    train_started = time.time()
    train_history = []

    for epoch in range(1, args.num_epochs + 1):
        model.reset_memory()
        loss, train_y, train_prob = _train_epoch(model, x, train_data, args)
        val_y, val_prob, val_eval_sec = _evaluate_split(model, x, [train_data], val_data, args)
        val_threshold, val_score = choose_threshold(val_y, val_prob, args.threshold_metric)
        train_history.append({
            "epoch": epoch,
            "loss": loss,
            "val_PR_AUC_negative": classification_metrics(val_y, val_prob, val_threshold)["PR_AUC_negative"],
            "val_F1_macro": classification_metrics(val_y, val_prob, val_threshold)["F1_macro"],
            "val_threshold": val_threshold,
            "val_threshold_score": val_score,
            "val_eval_sec": val_eval_sec,
        })

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= args.early_stop_patience:
            break

    train_sec = time.time() - train_started
    if best_state is not None:
        model.load_state_dict(best_state)

    val_y, val_prob, _ = _evaluate_split(model, x, [train_data], val_data, args)
    threshold, threshold_score = choose_threshold(val_y, val_prob, args.threshold_metric)
    test_y, test_prob, eval_sec = _evaluate_split(model, x, [train_data, val_data], test_data, args)

    metric_rows = []
    for threshold_mode, thr in [("val_F1_macro", threshold), ("fixed_0.5", 0.5)]:
        metrics = classification_metrics(test_y, test_prob, threshold=thr)
        metric_rows.append({
            **metrics,
            "dataset": args.dataset,
            "model": model_name,
            "feature_set": "graph_memory",
            "seed": seed,
            "threshold_mode": threshold_mode,
            "best_epoch": best_epoch,
            "val_threshold_score": threshold_score,
            "train_wall_time_sec": round(train_sec, 6),
            "eval_wall_time_sec": round(eval_sec, 6),
            "total_wall_time_sec": round(train_sec + eval_sec, 6),
            "device": args.device,
            "num_parameters": _model_param_count(model),
            "max_events_per_block": args.max_events_per_block,
        })

    prediction_rows = [{
        "dataset": args.dataset,
        "model": model_name,
        "seed": seed,
        "row_id": idx,
        "y_true": int(y),
        "y_prob": float(prob),
        "threshold": float(threshold),
        "y_pred": int(prob >= threshold),
    } for idx, (y, prob) in enumerate(zip(test_y, test_prob))]

    pd.DataFrame(train_history).to_csv(
        out_dir / "training_curves" / f"{model_name}_seed_{seed}.csv",
        index=False,
    )
    _append_csv(out_dir / "per_run_metrics.csv", metric_rows)
    _append_csv(out_dir / "per_run_predictions.csv", prediction_rows)
    return metric_rows


def _write_manifest(out_dir: Path, args, split_rows):
    manifest = {
        "runner": "run_graph_strict",
        "dataset": args.dataset,
        "processed_dir": args.processed_dir,
        "max_events": args.max_events,
        "models": args.models,
        "seeds": args.seeds,
        "num_epochs": args.num_epochs,
        "early_stop_patience": args.early_stop_patience,
        "max_events_per_block": args.max_events_per_block,
        "protocol_note": (
            "Predict-before-update no-leakage mini-block protocol. Identical "
            "timestamps are never split. Positive max_events_per_block can group "
            "multiple timestamps into a conservative mini-batch."
        ),
        "split_summary": split_rows,
    }
    (out_dir / "manifest_graph.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--results_dir", default="results/strict_compare_pilot")
    parser.add_argument("--models", nargs="+", choices=["tgn", "semba", "semba-noprop"], default=["tgn", "semba"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--max_events_per_block", type=int, default=1000)
    parser.add_argument("--threshold_metric", default="F1_macro")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feat_type", default="zeros")
    parser.add_argument("--num_feats", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr_init", type=float, default=0.01)
    parser.add_argument("--neg_wt", type=float, default=1.0)
    args = parser.parse_args()

    data, train_data, val_data, test_data = get_data(
        args.dataset,
        f"./data/{args.dataset}",
        args.device,
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )
    out_dir = Path(args.results_dir) / args.dataset
    (out_dir / "training_curves").mkdir(parents=True, exist_ok=True)
    split_rows = split_summary(data, train_data, val_data, test_data)
    _write_manifest(out_dir, args, split_rows)

    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            rows.extend(_run_one(model_name, seed, data, train_data, val_data, test_data, args, out_dir))
            print(json.dumps({
                "model": model_name,
                "seed": seed,
                "last_rows": rows[-2:],
            }, indent=2, ensure_ascii=False))

    print(json.dumps({
        "runner": "run_graph_strict",
        "out_dir": str(out_dir),
        "rows_written": len(rows),
        "models": args.models,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
