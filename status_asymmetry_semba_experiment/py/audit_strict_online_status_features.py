from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.strict_temporal_protocol import split_summary  # noqa: E402
from status_asymmetry_semba_experiment.py.build_online_status_features import (  # noqa: E402
    build_status_feature_frames,
    status_feature_audit_rows,
    write_feature_dictionary,
)
from utils import get_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "--data", default="BitcoinOTC-1")
    parser.add_argument("--processed_dir", default="processed2")
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--feature_set", default="all_status")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tables_dir", default="status_asymmetry_semba_experiment/tables")
    args = parser.parse_args()

    data, train_data, val_data, test_data = get_data(
        args.dataset,
        f"./data/{args.dataset}",
        args.device,
        processed_dir=args.processed_dir,
        max_events=args.max_events,
    )
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    frames = build_status_feature_frames(data, train_data, val_data, feature_set=args.feature_set)
    audit = status_feature_audit_rows(frames, train_data, val_data, test_data)
    pd.DataFrame(audit).to_csv(tables_dir / "status_feature_audit.csv", index=False)
    write_feature_dictionary(tables_dir / "status_feature_dictionary.csv")
    print(json.dumps({
        "dataset": args.dataset,
        "feature_set": args.feature_set,
        "audit_path": str(tables_dir / "status_feature_audit.csv"),
        "feature_dictionary": str(tables_dir / "status_feature_dictionary.csv"),
        "split_summary": split_summary(data, train_data, val_data, test_data),
        "audit": audit,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
