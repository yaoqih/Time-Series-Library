import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "stock_results_rolling"
REQUIRED_COLUMNS = {"code", "trade_date", "pred_return", "true_return", "suspendFlag"}

INITIAL_CASH = 1_000_000.0
COMMISSION = 0.0003  # 0.03%
STAMP_TAX = 0.001    # 0.1%
RISK_FREE = 0.03


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict
    curve_df: pd.DataFrame
    picks_df: pd.DataFrame
    daily_df: pd.DataFrame


def _format_date_yyyymmdd(raw: str) -> str:
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _window_label(name: str) -> str:
    pattern = (
        r"win(?P<win>\d+)_tr(?P<tr_s>\d{8})-(?P<tr_e>\d{8})_"
        r"te(?P<te_s>\d{8})-(?P<te_e>\d{8})_"
        r"va(?P<va_s>\d{8})-(?P<va_e>\d{8})"
    )
    match = re.match(pattern, name)
    if not match:
        return name
    g = match.groupdict()
    return (
        f"win{g['win']} | "
        f"train {_format_date_yyyymmdd(g['tr_s'])}~{_format_date_yyyymmdd(g['tr_e'])} | "
        f"test {_format_date_yyyymmdd(g['te_s'])}~{_format_date_yyyymmdd(g['te_e'])} | "
        f"val {_format_date_yyyymmdd(g['va_s'])}~{_format_date_yyyymmdd(g['va_e'])}"
    )


def _prepare_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(pred_df.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    df = pred_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.dropna(subset=["pred_return", "true_return"])
    df = df[np.isfinite(df["pred_return"]) & np.isfinite(df["true_return"])]
    df = df[df["true_return"] > -0.999]
    df = df[df["suspendFlag"] == 0]
    return df.sort_values("trade_date")


def _compute_sharpe(dates: pd.Series, capital: np.ndarray, initial_cash: float, risk_free: float) -> float:
    if len(capital) < 2:
        return 0.0
    total_days = (dates.iloc[-1] - dates.iloc[0]).days
    total_years = total_days / 365.25 if total_days > 0 else 0.0
    if total_years > 0 and initial_cash > 0 and capital[-1] > 0:
        annualized_return = (capital[-1] / initial_cash) ** (1 / total_years) - 1
    elif capital[-1] <= 0:
        annualized_return = -1.0
    else:
        annualized_return = 0.0
    daily_returns = np.diff(capital) / np.where(capital[:-1] > 0, capital[:-1], np.nan)
    daily_returns = np.nan_to_num(daily_returns, nan=0.0)
    annual_vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) else 0.0
    return (annualized_return - risk_free) / annual_vol if annual_vol > 0 else 0.0


def backtest_top1(
    pred_df: pd.DataFrame,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION,
    stamp: float = STAMP_TAX,
    risk_free: float = RISK_FREE,
) -> BacktestResult:
    if pred_df.empty:
        empty_metrics = {
            "final_capital": float(initial_cash),
            "cumulative_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "trade_days": 0,
        }
        empty = pd.DataFrame()
        return BacktestResult(empty_metrics, empty, empty, empty)

    df = _prepare_predictions(pred_df)
    if df.empty:
        return backtest_top1(pd.DataFrame(), initial_cash, commission, stamp, risk_free)

    cash = float(initial_cash)
    equity_curve = []
    trade_returns = []
    trade_profits = []
    picks_records = []

    total_profit = 0.0
    total_loss = 0.0

    for trade_date, group in df.groupby("trade_date", sort=True):
        pick = group.sort_values("pred_return", ascending=False).iloc[0]
        gross_return = float(pick["true_return"])
        net_factor = (1 - commission) * (1 + gross_return) * (1 - commission - stamp)
        if not np.isfinite(net_factor) or net_factor <= 0:
            equity_curve.append((trade_date, cash))
            continue
        net_return = net_factor - 1.0
        profit = cash * net_return
        cash *= net_factor
        trade_returns.append(net_return)
        trade_profits.append(profit)
        equity_curve.append((trade_date, cash))

        if profit > 0:
            total_profit += profit
        elif profit < 0:
            total_loss += -profit

        picks_records.append(
            {
                "trade_date": trade_date,
                "code": pick["code"],
                "pred_return": float(pick["pred_return"]),
                "true_return": float(pick["true_return"]),
                "net_return": float(net_return),
                "net_factor": float(net_factor),
                "capital": float(cash),
            }
        )

    curve_df = pd.DataFrame(equity_curve, columns=["date", "capital"])
    picks_df = pd.DataFrame(picks_records)

    if curve_df.empty:
        return backtest_top1(pd.DataFrame(), initial_cash, commission, stamp, risk_free)

    total_days = (curve_df["date"].iloc[-1] - curve_df["date"].iloc[0]).days if len(curve_df) > 1 else 0
    total_years = total_days / 365.25 if total_days > 0 else 0.0

    final_capital = float(curve_df["capital"].iloc[-1])
    cumulative_return = (final_capital - initial_cash) / initial_cash if initial_cash > 0 else 0.0
    if total_years > 0 and initial_cash > 0 and final_capital > 0:
        annualized_return = (final_capital / initial_cash) ** (1 / total_years) - 1
    elif final_capital <= 0:
        annualized_return = -1.0
    else:
        annualized_return = 0.0

    capital_values = curve_df["capital"].values
    peak = np.maximum.accumulate(capital_values)
    peak = np.where(peak > 0, peak, np.nan)
    drawdowns = (peak - capital_values) / peak
    max_drawdown = float(np.nanmax(drawdowns)) if len(drawdowns) else 0.0

    sharpe = _compute_sharpe(curve_df["date"], capital_values, initial_cash, risk_free)
    wins = [r for r in trade_returns if r > 0]
    win_rate = (len(wins) / len(trade_returns)) if trade_returns else 0.0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else float("inf") if total_profit > 0 else 0.0

    metrics = {
        "final_capital": final_capital,
        "cumulative_return_pct": float(cumulative_return * 100),
        "annualized_return_pct": float(annualized_return * 100),
        "max_drawdown_pct": float(max_drawdown * 100),
        "sharpe": float(sharpe),
        "win_rate_pct": float(win_rate * 100),
        "profit_factor": float(profit_factor),
        "total_trades": int(len(trade_returns) * 2),
        "trade_days": int(len(trade_returns)),
    }

    if not picks_df.empty:
        capital = picks_df["capital"].values
        picks_df["return_pct"] = (capital / initial_cash - 1.0) * 100.0
        peak = np.maximum.accumulate(capital)
        peak = np.where(peak > 0, peak, np.nan)
        drawdown = (peak - capital) / peak
        picks_df["drawdown_pct"] = drawdown * 100.0
        picks_df["max_drawdown_pct"] = np.maximum.accumulate(drawdown) * 100.0

    daily_df = picks_df.copy()
    daily_df.rename(columns={"trade_date": "date"}, inplace=True)

    return BacktestResult(metrics, curve_df, picks_df, daily_df)


@st.cache_data(show_spinner=False)
def list_models(results_root: Path) -> List[str]:
    if not results_root.exists():
        return []
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()])


@st.cache_data(show_spinner=False)
def list_windows(results_root: Path, model_name: str) -> List[Path]:
    model_dir = results_root / model_name
    if not model_dir.exists():
        return []

    def sort_key(p: Path):
        match = re.search(r"win(\d+)", p.name)
        if match:
            return (0, int(match.group(1)))
        return (1, p.name)

    return sorted([p for p in model_dir.iterdir() if p.is_dir()], key=sort_key)


@st.cache_data(show_spinner=False)
def list_search_runs(results_root: Path) -> List[str]:
    if not results_root.exists():
        return []
    runs = []
    for item in results_root.iterdir():
        if not item.is_dir():
            continue
        search_summary = item / "search" / "combo_summary.csv"
        if search_summary.exists():
            runs.append(item.name)
    return sorted(runs)


@st.cache_data(show_spinner=False)
def load_window_summary(results_root: Path, model_name: str) -> pd.DataFrame:
    summary_path = results_root / model_name / "window_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


@st.cache_data(show_spinner=False)
def load_combo_summary(results_root: Path, run_name: str) -> pd.DataFrame:
    summary_path = results_root / run_name / "search" / "combo_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_path)


@st.cache_data(show_spinner=False)
def load_combo_details(results_root: Path, run_name: str) -> List[Dict]:
    details_path = results_root / run_name / "search" / "combo_details.json"
    if not details_path.exists():
        return []
    with open(details_path, "r") as f:
        return json.load(f)


def _extract_split_prefixes(summary_df: pd.DataFrame) -> List[str]:
    prefixes = set()
    for col in summary_df.columns:
        if "_" not in col:
            continue
        prefix, _ = col.split("_", 1)
        if prefix in {"test", "val", "train"}:
            prefixes.add(prefix)
    return sorted(prefixes)


def _metric_stats(summary_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    cols = [c for c in numeric_cols if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    records = []
    for col in cols:
        series = summary_df[col].dropna()
        if series.empty:
            continue
        records.append({
            "metric": col[len(prefix):],
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        })
    return pd.DataFrame(records).sort_values("metric")


def _simple_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    records = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        records.append({
            "metric": col,
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        })
    return pd.DataFrame(records).sort_values("metric")

@st.cache_data(show_spinner=False)
def _compute_backtest_cached(pred_path: str, mtime: float):
    pred_df = pd.read_csv(pred_path)
    result = backtest_top1(pred_df)
    return result.metrics, result.curve_df, result.picks_df, result.daily_df


def compute_backtest(pred_path: str, mtime: float) -> BacktestResult:
    metrics, curve_df, picks_df, daily_df = _compute_backtest_cached(pred_path, mtime)
    return BacktestResult(metrics, curve_df, picks_df, daily_df)


def _render_metrics(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Capital", f"{metrics['final_capital']:,.2f}")
    c2.metric("Cumulative Return", f"{metrics['cumulative_return_pct']:.2f}%")
    c3.metric("Annualized Return", f"{metrics['annualized_return_pct']:.2f}%")
    c4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sharpe", f"{metrics['sharpe']:.3f}")
    c6.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
    c7.metric("Profit Factor", f"{metrics['profit_factor']:.3f}")
    c8.metric("Trades", f"{metrics['total_trades']} (days {metrics['trade_days']})")


def _render_charts(result: BacktestResult):
    if result.daily_df.empty:
        st.info("该窗口无可用交易记录。")
        return

    curve = result.daily_df[["date", "capital", "return_pct", "drawdown_pct"]].copy()
    curve = curve.set_index("date")
    left, right = st.columns(2)
    with left:
        st.line_chart(curve[["capital"]], height=260)
        st.caption("资金曲线（计入佣金和印花税）")
    with right:
        st.line_chart(curve[["return_pct", "drawdown_pct"]], height=260)
        st.caption("收益率与回撤（%）")


def _render_trades(result: BacktestResult):
    if result.picks_df.empty:
        return
    st.subheader("交易股票与明细")
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(
            result.picks_df[[
                "trade_date", "code", "pred_return", "true_return", "net_return", "capital"
            ]],
            height=360,
            width="stretch",
        )
    with right:
        top_codes = result.picks_df["code"].value_counts().head(12)
        st.bar_chart(top_codes, height=360)
        st.caption("交易频次 Top 12")


def _render_window_overview(summary_df: pd.DataFrame):
    st.subheader("滚动回测汇总")
    if summary_df.empty:
        st.info("未找到 window_summary.csv，可先运行滚动回测生成汇总。")
        return

    prefixes = _extract_split_prefixes(summary_df)
    if not prefixes:
        st.info("汇总表中未检测到 test_/val_ 前缀指标。")
        return

    split = st.selectbox("指标分组", prefixes, index=0, key="summary_split")
    prefix = f"{split}_"
    stats_df = _metric_stats(summary_df, prefix)
    if stats_df.empty:
        st.info("当前分组无数值指标。")
        return

    metrics = stats_df["metric"].tolist()
    default_metrics = [m for m in ["cumulative_return_pct", "annualized_return_pct", "max_drawdown_pct", "sharpe"] if m in metrics]
    if not default_metrics:
        default_metrics = metrics[:3]
    selected_metrics = st.multiselect(
        "窗口曲线指标",
        metrics,
        default=default_metrics,
        key="summary_metrics"
    )

    left, right = st.columns([2, 1])
    with left:
        if "window_tag" in summary_df.columns and selected_metrics:
            plot_cols = [f"{prefix}{m}" for m in selected_metrics if f"{prefix}{m}" in summary_df.columns]
            if plot_cols:
                plot_df = summary_df.set_index("window_tag")[plot_cols].copy()
                plot_df.columns = [c[len(prefix):] for c in plot_cols]
                st.line_chart(plot_df, height=280)
        st.caption(f"窗口数：{len(summary_df)}")
        st.dataframe(summary_df, height=320, width="stretch")
    with right:
        st.markdown("**指标均值/波动**")
        st.dataframe(stats_df, height=320, width="stretch")

        sort_metric = st.selectbox("排序指标", metrics, index=0, key="summary_sort_metric")
        if "window_tag" in summary_df.columns and f"{prefix}{sort_metric}" in summary_df.columns:
            ranked = summary_df[["window_tag", f"{prefix}{sort_metric}"]].sort_values(
                f"{prefix}{sort_metric}", ascending=False
            )
            st.markdown("**Top 5 窗口**")
            st.dataframe(ranked.head(5), height=200, width="stretch")
            st.markdown("**Bottom 5 窗口**")
            st.dataframe(ranked.tail(5), height=200, width="stretch")


def _render_search_overview(combo_summary: pd.DataFrame, combo_details: List[Dict]):
    st.subheader("策略搜索汇总")
    if combo_summary.empty:
        st.info("未找到策略搜索汇总（combo_summary.csv）。")
        return

    filtered = combo_summary.copy()
    if "size" in filtered.columns:
        sizes = sorted({int(s) for s in filtered["size"].dropna().tolist()})
        if sizes:
            size_pick = st.multiselect("组合大小", sizes, default=sizes, key="search_sizes")
            if size_pick:
                filtered = filtered[filtered["size"].isin(size_pick)]

    metric_cols = [c for c in filtered.columns if c.startswith("mean_test_")]
    if metric_cols:
        default_metric = "mean_test_cumulative_return_pct" if "mean_test_cumulative_return_pct" in metric_cols else metric_cols[0]
        sort_metric = st.selectbox("排序指标", metric_cols, index=metric_cols.index(default_metric), key="search_sort")
        filtered = filtered.sort_values(sort_metric, ascending=False)

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(filtered, height=360, width="stretch")
    with right:
        if metric_cols:
            st.markdown("**Top 10 组合**")
            cols = ["models", "size", "num_windows", sort_metric]
            cols = [c for c in cols if c in filtered.columns]
            st.dataframe(filtered.head(10)[cols], height=360, width="stretch")

    if not combo_details:
        return
    combo_map = {",".join(item.get("models", [])): item for item in combo_details if item.get("models")}
    if not combo_map:
        return

    st.markdown("**组合窗口明细**")
    combo_key = st.selectbox("选择组合", sorted(combo_map.keys()), key="search_combo_pick")
    detail = combo_map[combo_key]
    window_metrics = detail.get("window_metrics") or {}
    if window_metrics:
        rows = []
        for window_tag, metrics in window_metrics.items():
            row = {"window_tag": window_tag}
            if isinstance(metrics, dict):
                row.update(metrics)
            else:
                row["cumulative_return_pct"] = metrics
            rows.append(row)
        window_df = pd.DataFrame(rows).sort_values("window_tag")
        metrics_stats = _simple_stats(window_df.drop(columns=["window_tag"], errors="ignore"))
        left, right = st.columns([2, 1])
        with left:
            st.dataframe(window_df, height=320, width="stretch")
        with right:
            st.markdown("**窗口指标均值**")
            st.dataframe(metrics_stats, height=320, width="stretch")
def _load_split(window_dir: Path, split: str) -> Optional[BacktestResult]:
    pred_path = window_dir / split / "predictions.csv"
    if not pred_path.exists():
        return None
    mtime = pred_path.stat().st_mtime
    return compute_backtest(str(pred_path), mtime)


def main():
    st.set_page_config(page_title="Rolling Stock Backtest", layout="wide")
    st.title("滚动回测结果可视化")
    st.caption("依据 predictions.csv：每日选取未停牌且 pred_return 最高的股票，用 true_return 计算真实收益。")

    models = list_models(RESULTS_ROOT)
    if not models:
        st.error(f"未找到结果目录：{RESULTS_ROOT}")
        return

    search_runs = list_search_runs(RESULTS_ROOT)

    with st.sidebar:
        st.header("筛选条件")
        model = st.selectbox("模型", models, index=0)
        windows = list_windows(RESULTS_ROOT, model)
        window = None
        if not windows:
            st.warning("该模型没有窗口数据。")
        else:
            window = st.selectbox("时间窗口", windows, format_func=lambda p: _window_label(p.name))
        st.markdown("**参数**")
        st.write(f"初始资金：{INITIAL_CASH:,.0f}")
        st.write(f"佣金：{COMMISSION*100:.2f}%")
        st.write(f"印花税：{STAMP_TAX*100:.2f}%")
        if search_runs:
            st.divider()
            st.subheader("策略搜索")
            search_run = st.selectbox("搜索结果", search_runs, index=0)
        else:
            search_run = None

    overview_tab, detail_tab = st.tabs(["概览", "窗口回测"])
    with overview_tab:
        summary_df = load_window_summary(RESULTS_ROOT, model)
        _render_window_overview(summary_df)
        st.divider()
        if search_run:
            combo_summary = load_combo_summary(RESULTS_ROOT, search_run)
            combo_details = load_combo_details(RESULTS_ROOT, search_run)
            _render_search_overview(combo_summary, combo_details)
        else:
            st.subheader("策略搜索汇总")
            st.info("未检测到策略搜索结果目录。")

    with detail_tab:
        if not window:
            st.info("请先选择包含窗口数据的模型。")
            return
        split_tabs = st.tabs(["Test", "Val"])
        for split, tab in zip(["test", "val"], split_tabs):
            with tab:
                result = _load_split(window, split)
                if result is None:
                    st.info(f"{split} 无预测文件。")
                    continue
                _render_metrics(result.metrics)
                _render_charts(result)
                _render_trades(result)


if __name__ == "__main__":
    main()
