from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.strict_metrics import choose_threshold, classification_metrics
from experiments.strict_temporal_protocol import split_summary
from model_wrapper import STGNN
from utils import LRScheduler, get_data, seq_batches


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


def _edge_parts(batch, device: str):
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


def _forward_update_first(model, x, parts):
    return model(
        x,
        parts["pos_edge_index"],
        parts["neg_edge_index"],
        parts["pos_times"],
        parts["neg_times"],
        parts["pos_weights"],
        parts["neg_weights"],
        to_update=True,
    )


def _metrics_with_aliases(y_true, y_prob, threshold: float):
    metrics = classification_metrics(y_true, y_prob, threshold)
    metrics["F1_bin"] = metrics["F1_positive"]
    metrics["TN"] = metrics["true_negative"]
    metrics["FP"] = metrics["false_positive"]
    metrics["FN"] = metrics["false_negative"]
    metrics["TP"] = metrics["true_positive"]
    return metrics


def _model_param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _append_csv(path: Path, rows) -> None:
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path, low_memory=False)
        pd.concat([existing, df], ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def _train_epoch(model, x, train_data, optimizer, args):
    model.train()
    total_loss = 0.0
    total_examples = 0
    all_y = []
    all_prob = []

    for batch in seq_batches(train_data, batch_size=args.batch_size):
        optimizer.zero_grad()
        parts = _edge_parts(batch, args.device)
        z = _forward_update_first(model, x, parts)
        loss = model.loss(
            z,
            parts["pos_edge_index"],
            parts["neg_edge_index"],
            neg_wt=args.neg_wt,
        )
        prob = model.predict(z, parts["src"], parts["dst"]).detach()
        target = parts["signs"].detach()
        n = int(target.numel())
        total_loss += float(loss.detach()) * n
        total_examples += n
        all_y.extend(target.cpu().numpy().astype(int).tolist())
        all_prob.extend(prob.cpu().numpy().tolist())
        loss.backward()
        optimizer.step()

    return total_loss / max(1, total_examples), all_y, all_prob


@torch.no_grad()
def _evaluate_update_first(model, x, eval_data, args):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_y = []
    all_prob = []
    started = time.time()

    for batch in seq_batches(eval_data, batch_size=args.batch_size):
        parts = _edge_parts(batch, args.device)
        z = _forward_update_first(model, x, parts)
        loss = model.loss(
            z,
            parts["pos_edge_index"],
            parts["neg_edge_index"],
            neg_wt=args.neg_wt,
        )
        prob = model.predict(z, parts["src"], parts["dst"]).detach()
        target = parts["signs"].detach()
        n = int(target.numel())
        total_loss += float(loss.detach()) * n
        total_examples += n
        all_y.extend(target.cpu().numpy().astype(int).tolist())
        all_prob.extend(prob.cpu().numpy().tolist())

    return total_loss / max(1, total_examples), all_y, all_prob, time.time() - started


def _run_one(model_name: str, seed: int, data, train_data, val_data, test_data, args, out_dir: Path):
    _set_seed(seed)
    x, num_feats = _make_node_features(data, args.feat_type, args.num_feats, args.device)
    model = STGNN(
        model_name,
        args.task,
        num_feats,
        data.num_nodes,
        args.embedding_dim,
        args.num_layers,
        device=args.device,
        debug=False,
    )
    model.to(args.device)
    optimizer = torch.optim.Adam(set(model.parameters()), lr=args.lr_init)
    lr_scheduler = LRScheduler(optimizer)

    train_started = time.time()
    train_history = []
    final_val_y = []
    final_val_prob = []

    for epoch in range(1, args.num_epochs + 1):
        model.reset_memory()
        train_loss, train_y, train_prob = _train_epoch(model, x, train_data, optimizer, args)
        val_loss, val_y, val_prob, val_eval_sec = _evaluate_update_first(model, x, val_data, args)
        lr_scheduler(val_loss)
        final_val_y = val_y
        final_val_prob = val_prob
        fixed_val_metrics = _metrics_with_aliases(val_y, val_prob, 0.5)
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_AUROC": fixed_val_metrics["AUROC"],
            "val_F1_bin_fixed_0.5": fixed_val_metrics["F1_bin"],
            "val_F1_macro_fixed_0.5": fixed_val_metrics["F1_macro"],
            "val_PR_AUC_negative": fixed_val_metrics["PR_AUC_negative"],
            "val_eval_sec": val_eval_sec,
        })

    train_sec = time.time() - train_started
    val_f1_bin_threshold, val_f1_bin_score = choose_threshold(
        final_val_y,
        final_val_prob,
        "F1_positive",
    )
    val_f1_macro_threshold, val_f1_macro_score = choose_threshold(
        final_val_y,
        final_val_prob,
        "F1_macro",
    )

    test_loss, test_y, test_prob, eval_sec = _evaluate_update_first(model, x, test_data, args)
    metric_rows = []
    for threshold_mode, threshold, val_score in [
        ("fixed_0.5", 0.5, np.nan),
        ("val_F1_bin", val_f1_bin_threshold, val_f1_bin_score),
        ("val_F1_macro", val_f1_macro_threshold, val_f1_macro_score),
    ]:
        metrics = _metrics_with_aliases(test_y, test_prob, threshold)
        metric_rows.append({
            **metrics,
            "dataset": args.dataset,
            "model": model_name,
            "feature_set": "graph_memory",
            "seed": seed,
            "threshold_mode": threshold_mode,
            "val_threshold_score": val_score,
            "train_wall_time_sec": round(train_sec, 6),
            "eval_wall_time_sec": round(eval_sec, 6),
            "total_wall_time_sec": round(train_sec + eval_sec, 6),
            "device": args.device,
            "num_parameters": _model_param_count(model),
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr_init": args.lr_init,
            "embedding_dim": args.embedding_dim,
            "test_loss": test_loss,
        })

    prediction_rows = [{
        "dataset": args.dataset,
        "model": model_name,
        "feature_set": "graph_memory",
        "seed": seed,
        "row_id": idx,
        "y_true": int(y),
        "y_prob": float(prob),
        "threshold_fixed_0.5": 0.5,
        "threshold_val_F1_bin": float(val_f1_bin_threshold),
        "threshold_val_F1_macro": float(val_f1_macro_threshold),
        "y_pred_fixed_0.5": int(prob >= 0.5),
        "y_pred_val_F1_bin": int(prob >= val_f1_bin_threshold),
        "y_pred_val_F1_macro": int(prob >= val_f1_macro_threshold),
    } for idx, (y, prob) in enumerate(zip(test_y, test_prob))]

    pd.DataFrame(train_history).to_csv(
        out_dir / "training_curves" / f"{model_name}_seed_{seed}_emb{args.embedding_dim}.csv",
        index=False,
    )
    _append_csv(out_dir / "per_run_metrics.csv", metric_rows)
    _append_csv(out_dir / "per_run_predictions.csv", prediction_rows)
    return metric_rows


def _write_manifest(out_dir: Path, args, split_rows) -> None:
    manifest = {
        "runner": "run_graph_paper_protocol",
        "task": args.task,
        "protocol": "paper_update_before_predict_to_update_true",
        "leakage_note": (
            "This runner intentionally matches the old train.py style: each "
            "batch is written into SEMBA/TGN memory before the same batch is "
            "predicted. This is paper-protocol comparison, not strict online "
            "no-leakage evaluation."
        ),
        "dataset": args.dataset,
        "processed_dir": args.processed_dir,
        "max_events": args.max_events,
        "models": args.models,
        "seeds": args.seeds,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr_init": args.lr_init,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "feat_type": args.feat_type,
        "num_feats": args.num_feats,
        "device": args.device,
        "split_summary": split_rows,
    }
    (out_dir / "manifest_graph_paper_protocol.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--results_dir", default="paper_protocol_sign_class_experiment/results")
    parser.add_argument("--models", nargs="+", choices=["tgn", "semba", "semba-noprop"], default=["tgn", "semba"])
    parser.add_argument("--task", default="sign_class", choices=["sign_class"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feat_type", default="zeros")
    parser.add_argument("--num_feats", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr_init", type=float, default=0.01)
    parser.add_argument("--neg_wt", type=float, default=1.0)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

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
            new_rows = _run_one(model_name, seed, data, train_data, val_data, test_data, args, out_dir)
            rows.extend(new_rows)
            print(json.dumps({
                "runner": "run_graph_paper_protocol",
                "model": model_name,
                "seed": seed,
                "rows_written_for_seed": len(new_rows),
            }, indent=2, ensure_ascii=False))

    print(json.dumps({
        "runner": "run_graph_paper_protocol",
        "out_dir": str(out_dir),
        "rows_written": len(rows),
        "models": args.models,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
