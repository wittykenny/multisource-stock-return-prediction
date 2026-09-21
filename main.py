from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stock_predictor import StockPredictionPipeline, load_config
from stock_predictor.utils import serialize_float_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-source heterogeneous stock return prediction pipeline")
    parser.add_argument("--config", default=str(Path("config") / "default.yaml"), help="Path to YAML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("fetch", help="Fetch and cache raw data")
    subparsers.add_parser("prepare", help="Build merged modeling dataset")
    subparsers.add_parser("backtest", help="Run rolling-window backtest")
    subparsers.add_parser("audit", help="Prepare data and write baseline leakage audit")
    subparsers.add_parser("tune", help="Run main-model hyperparameter search without baselines")
    subparsers.add_parser("multiseed", help="Run main-model multi-seed robustness tests without baselines")
    subparsers.add_parser("ablation", help="Run full model and ablation backtests")
    subparsers.add_parser("ablation-significance", help="Compare full ablation against each ablated variant")
    subparsers.add_parser("info-gain", help="Compute mutual-information feature and modality utility scores")
    subparsers.add_parser("full", help="Run fetch + prepare + backtest")
    return parser


def main() -> None:
    _configure_console_encoding()
    args = build_parser().parse_args()
    config = load_config(args.config)
    pipeline = StockPredictionPipeline(config)
    if args.command == "fetch":
        _safe_print(pipeline.run_fetch())
    elif args.command == "prepare":
        dataset = pipeline.run_prepare()
        _safe_print({"rows": len(dataset.frame), "numeric_features": len(dataset.numeric_features), "text_features": len(dataset.text_features), "metadata": dataset.metadata})
    elif args.command == "backtest":
        _safe_print(pipeline.run_backtest())
    elif args.command == "audit":
        _safe_print(pipeline.run_leakage_audit())
    elif args.command == "tune":
        _safe_print(pipeline.run_hyperparameter_search())
    elif args.command == "multiseed":
        _safe_print(pipeline.run_multi_seed())
    elif args.command == "ablation":
        _safe_print(pipeline.run_ablation())
    elif args.command == "ablation-significance":
        _safe_print(pipeline.run_ablation_significance())
    elif args.command == "info-gain":
        _safe_print(pipeline.run_information_gain())
    elif args.command == "full":
        _safe_print(pipeline.run_full())


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _safe_print(value: object) -> None:
    payload = serialize_float_map(value)
    try:
        print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(payload, ensure_ascii=True, default=str, indent=2))


if __name__ == "__main__":
    main()
