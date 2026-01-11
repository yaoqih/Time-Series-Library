import argparse
import importlib.util
import json
import os
import sys

import pandas as pd


def _load_backtest_topk_detailed():
    here = os.path.dirname(__file__)
    module_fp = os.path.join(here, "stock_rolling_backtest.py")
    spec = importlib.util.spec_from_file_location("stock_rolling_backtest", module_fp)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_fp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "backtest_topk_detailed", None)
    if fn is None:
        raise ImportError("stock_rolling_backtest.py missing backtest_topk_detailed")
    return fn


def parse_args():
    parser = argparse.ArgumentParser(description="Recalculate rolling stock metrics from existing predictions.csv")

    parser.add_argument("--results_root", type=str, default="stock_results_rolling", help="rolling results root dir")
    parser.add_argument("--models", type=str, default="", help="comma-separated model dirs; empty = all")
    parser.add_argument("--splits", type=str, default="test,val", help="comma-separated splits")
    parser.add_argument("--window_glob", type=str, default="", help="glob to filter window_tag dirs (e.g., win007_*)")

    parser.add_argument("--initial_cash", type=float, default=1_000_000.0, help="initial capital")
    parser.add_argument("--topk", type=int, default=1, help="topk per day")
    parser.add_argument("--commission", type=float, default=0.0003, help="commission rate")
    parser.add_argument("--stamp", type=float, default=0.001, help="stamp tax rate")
    parser.add_argument("--risk_free", type=float, default=0.03, help="risk free rate")
    parser.add_argument(
        "--nan_true_return_policy",
        type=str,
        default="drop",
        choices=["drop", "zero"],
        help="how to handle NaN true_return rows",
    )

    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite existing metrics outputs")
    parser.add_argument("--dry_run", action="store_true", default=False, help="print planned updates only")
    parser.add_argument("--limit", type=int, default=0, help="max split dirs to process (0 = no limit)")
    return parser.parse_args()


def _iter_dirs(root: str):
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        fp = os.path.join(root, name)
        if os.path.isdir(fp):
            yield name, fp


def _iter_window_dirs(model_dir: str, window_glob: str):
    if not os.path.isdir(model_dir):
        return
    if window_glob:
        import glob

        for fp in sorted(glob.glob(os.path.join(model_dir, window_glob))):
            if os.path.isdir(fp):
                yield os.path.basename(fp), fp
        return
    yield from _iter_dirs(model_dir)


def main():
    args = parse_args()
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    model_filter = {m.strip() for m in str(args.models).split(",") if m.strip()}

    backtest_topk_detailed = _load_backtest_topk_detailed()
    processed = 0

    for model_name, model_dir in _iter_dirs(args.results_root):
        if model_filter and model_name not in model_filter:
            continue
        for window_tag, window_dir in _iter_window_dirs(model_dir, args.window_glob):
            for split in splits:
                split_dir = os.path.join(window_dir, split)
                pred_fp = os.path.join(split_dir, "predictions.csv")
                if not os.path.exists(pred_fp):
                    continue

                out_suffix = "" if args.overwrite else "_recalc"
                out_files = {
                    "equity_curve": os.path.join(split_dir, f"equity_curve{out_suffix}.csv"),
                    "daily_picks": os.path.join(split_dir, f"daily_picks{out_suffix}.csv"),
                    "daily_metrics": os.path.join(split_dir, f"daily_metrics{out_suffix}.csv"),
                    "cs_diag": os.path.join(split_dir, f"cross_sectional_diagnostics{out_suffix}.csv"),
                    "metrics": os.path.join(split_dir, f"metrics{out_suffix}.json"),
                }

                processed += 1
                print(f"[{processed}] {model_name}/{window_tag}/{split}: {pred_fp}")
                if args.dry_run:
                    continue

                try:
                    pred_df = pd.read_csv(pred_fp)
                    metrics, curve_df, picks_df, daily_df, cs_diag_df = backtest_topk_detailed(
                        pred_df,
                        initial_cash=args.initial_cash,
                        topk=args.topk,
                        commission=args.commission,
                        stamp=args.stamp,
                        risk_free=args.risk_free,
                        nan_true_return_policy=args.nan_true_return_policy,
                    )
                except Exception as exc:
                    print(f"[warn] failed to compute metrics for {pred_fp}: {exc}")
                    continue

                try:
                    curve_df.to_csv(out_files["equity_curve"], index=False)
                    picks_df.to_csv(out_files["daily_picks"], index=False)
                    daily_df.to_csv(out_files["daily_metrics"], index=False)
                    if cs_diag_df is not None:
                        cs_diag_df.to_csv(out_files["cs_diag"], index=False)
                    with open(out_files["metrics"], "w") as f:
                        json.dump(metrics, f, indent=2)
                except Exception as exc:
                    print(f"[warn] failed to write outputs under {split_dir}: {exc}")
                    continue

                if args.limit and processed >= int(args.limit):
                    return


if __name__ == "__main__":
    main()

