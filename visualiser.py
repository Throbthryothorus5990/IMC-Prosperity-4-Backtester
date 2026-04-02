"""
Visualiser for the Python IMC Prosperity 4 backtester.

Produces:
  1. PnL chart (total + per-product)
  2. Max drawdown chart
  3. Mid-price + inventory overlay (dual axis)
  4. Return distribution histogram
  5. Full metrics table

Designed for both terminal and Google Colab (inline matplotlib).
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from backtester import BacktestResult, SHORT_PRODUCT_LABELS


def _save_and_show(fig, save_path: str, label: str = "") -> None:
    """Save chart to file, then show inline (if interactive) or print path."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"  [SAVED] {save_path}")
    plt.show()  # Shows inline in Colab/Jupyter (%matplotlib inline), no-op in Agg
    plt.close(fig)



# ─── Metrics Computation ─────────────────────────────────────────────────────


def compute_metrics(result: BacktestResult) -> Dict[str, Any]:
    """Compute all backtest performance metrics from a BacktestResult."""

    pnl_values = [entry["total"] for entry in result.pnl_series]
    timestamps = [entry["timestamp"] for entry in result.pnl_series]

    if len(pnl_values) < 2:
        return _empty_metrics(result)

    # ── Drawdown calculations ──
    cummax = []
    running_max = -math.inf
    for v in pnl_values:
        running_max = max(running_max, v)
        cummax.append(running_max)

    drawdowns = [pnl_values[i] - cummax[i] for i in range(len(pnl_values))]
    max_drawdown = min(drawdowns) if drawdowns else 0.0

    # ── Profit Factor ──
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(1, len(pnl_values)):
        diff = pnl_values[i] - pnl_values[i - 1]
        if diff > 0:
            gross_profit += diff
        elif diff < 0:
            gross_loss += abs(diff)
            
    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # ── Returns ──
    returns = []
    for i in range(1, len(pnl_values)):
        returns.append(pnl_values[i] - pnl_values[i - 1])
    returns_arr = np.array(returns) if returns else np.array([0.0])

    # ── Sharpe ratio ──
    # Note: High-Frequency algorithmic trading has exceptionally tiny standard deviations
    # on tick returns. If we strictly annualized by 2,520,000 ticks/year, the Sharpe 
    # would be 80+. We scale by standard sqrt(252) to give a human-readable normalized 
    # ratio that bounds exactly as standard finance expects (1.0 to 5.0).
    annualization_factor = math.sqrt(252)
    mean_return = float(np.mean(returns_arr))
    std_return = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 1e-10
    sharpe_ratio = (mean_return / std_return * annualization_factor) if std_return > 1e-10 else 0.0

    # ── Calmar ratio ──
    total_return = pnl_values[-1] - pnl_values[0]
    calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # ── Trade statistics ──
    total_trades = result.metrics.own_trade_count
    fills = [t.quantity for t in result.own_trades_all]
    avg_fill = sum(fills) / len(fills) if fills else 0.0

    # ── Win rate (based on per-trade PnL contribution against fair value) ──
    enriched = getattr(result, "own_trades_enriched", [])
    if enriched:
        winning = sum(1 for t in enriched if t["pnl_contribution"] > 0)
        win_rate = winning / len(enriched)
        winning_trades = winning
    else:
        win_rate = 0.0
        winning_trades = 0

    # ── Recovery ──
    # Time from max-drawdown trough to recovery (new high)
    if max_drawdown < 0:
        trough_idx = drawdowns.index(max_drawdown)
        recovery_idx = None
        for i in range(trough_idx + 1, len(pnl_values)):
            if pnl_values[i] >= cummax[trough_idx]:
                recovery_idx = i
                break
        if recovery_idx is not None:
            recovery_ticks = recovery_idx - trough_idx
            recovery_str = f"{recovery_ticks} ticks"
        else:
            recovery_str = "Not recovered"
    else:
        recovery_str = "No drawdown"

    return {
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "total_trades": total_trades,
        "avg_fill": avg_fill,
        "recovery": recovery_str,
        "final_pnl": pnl_values[-1],
        "tick_count": result.metrics.tick_count,
        "drawdowns": drawdowns,
        "returns": returns,
        "pnl_values": pnl_values,
        "timestamps": timestamps,
        "cummax": cummax,
        "win_rate": win_rate,
        "winning_trades": winning_trades,
    }

def _empty_metrics(result: BacktestResult) -> Dict[str, Any]:
    return {
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "calmar_ratio": 0.0,
        "total_trades": result.metrics.own_trade_count,
        "avg_fill": 0.0,
        "recovery": "N/A",
        "final_pnl": 0.0,
        "tick_count": result.metrics.tick_count,
        "drawdowns": [],
        "returns": [],
        "pnl_values": [],
        "timestamps": [],
        "cummax": [],
        "win_rate": 0.0,
        "winning_trades": 0,
    }


# ─── Product-wise Metrics ────────────────────────────────────────────────────


def compute_product_metrics(
    result: BacktestResult,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-product PnL metrics."""
    product_metrics: Dict[str, Dict[str, Any]] = {}

    for product in result.products:
        pnl_vals = [entry.get(product, 0.0) for entry in result.pnl_series]
        if len(pnl_vals) < 2:
            product_metrics[product] = {
                "final_pnl": pnl_vals[-1] if pnl_vals else 0.0,
                "max_drawdown": 0.0,
                "trades": 0,
            }
            continue

        cummax = []
        running_max = -math.inf
        for v in pnl_vals:
            running_max = max(running_max, v)
            cummax.append(running_max)
        drawdowns = [pnl_vals[i] - cummax[i] for i in range(len(pnl_vals))]
        max_dd = min(drawdowns) if drawdowns else 0.0

        trade_count = sum(
            1 for t in result.own_trades_all if t.symbol == product
        )

        product_metrics[product] = {
            "final_pnl": pnl_vals[-1],
            "max_drawdown": max_dd,
            "trades": trade_count,
        }

    return product_metrics


# ─── Metrics Printing ────────────────────────────────────────────────────────


def print_metrics(metrics: Dict[str, Any], label: str = "") -> None:
    """Print formatted metrics to stdout."""
    header = f"═══ METRICS{f' — {label}' if label else ''} ═══"
    print(f"\n{'═' * len(header)}")
    print(header)
    print(f"{'═' * len(header)}")
    print(f"  Final PnL:          {metrics['final_pnl']:>12.2f}")
    print(f"  Tick Count:         {metrics['tick_count']:>12d}")
    print(f"  Total Trades:       {metrics['total_trades']:>12d}")
    print(f"  Avg Fill:           {metrics['avg_fill']:>12.2f}")
    wr = metrics.get("win_rate", 0.0)
    wt = metrics.get("winning_trades", 0)
    print(f"  Win Rate:           {wr*100:>11.1f}%  ({wt}/{metrics['total_trades']} trades)")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:>12.2f}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>12.4f}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>12.4f}")
    print(f"  Profit Factor:      {metrics['profit_factor']:>12.4f}")
    print(f"  Recovery:           {metrics['recovery']:>12s}")
    print(f"{'═' * len(header)}")


def print_product_table(
    product_metrics: Dict[str, Dict[str, Any]], label: str = ""
) -> None:
    """Print per-product PnL table."""
    header = f"─── Product Analysis{f' — {label}' if label else ''} ───"
    print(f"\n{header}")
    print(f"  {'PRODUCT':<25s} {'FINAL_PNL':>12s} {'MAX_DD':>12s} {'TRADES':>8s}")
    for product, pm in product_metrics.items():
        short = SHORT_PRODUCT_LABELS.get(product, product)
        print(
            f"  {short:<25s} {pm['final_pnl']:>12.2f} "
            f"{pm['max_drawdown']:>12.2f} {pm['trades']:>8d}"
        )
    print(f"{'─' * len(header)}")


# ─── Visualisations ──────────────────────────────────────────────────────────


def visualise(
    result: BacktestResult,
    label: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generate all visualisations for a backtest result:
      1. PnL timeseries (total + per-product)
      2. Drawdown chart
      3. Mid-price + Inventory overlay
      4. Return distribution
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed — skipping visualisations.")
        print("       Install with: pip install matplotlib numpy")
        return

    metrics = compute_metrics(result)
    product_metrics = compute_product_metrics(result)

    if not metrics["pnl_values"]:
        print("[WARN] No PnL data to visualise.")
        return

    # ── Style setup ──
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")

    title = f"Backtest Analysis{f' — {label}' if label else ''}"
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#58a6ff", y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    timestamps = metrics["timestamps"]
    pnl_values = metrics["pnl_values"]

    # Color palette
    colors = [
        "#58a6ff", "#3fb950", "#f0883e", "#d2a8ff",
        "#ff7b72", "#79c0ff", "#7ee787", "#ffa657",
    ]

    # ═══ Panel 1: PnL Chart ═══
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    ax1.plot(
        timestamps, pnl_values, color="#58a6ff", linewidth=1.8,
        label="Total PnL", zorder=5,
    )
    ax1.fill_between(
        timestamps, 0, pnl_values, alpha=0.15, color="#58a6ff",
    )
    # Per-product lines
    for i, product in enumerate(result.products):
        prod_pnl = [entry.get(product, 0.0) for entry in result.pnl_series]
        color = colors[(i + 1) % len(colors)]
        short = SHORT_PRODUCT_LABELS.get(product, product)
        ax1.plot(
            timestamps, prod_pnl, color=color, linewidth=0.9,
            alpha=0.7, label=short,
        )
    ax1.axhline(y=0, color="#484f58", linewidth=0.5, linestyle="--")
    ax1.set_title("Profit & Loss", fontsize=13, color="#c9d1d9", pad=10)
    ax1.set_xlabel("Timestamp", fontsize=10, color="#8b949e")
    ax1.set_ylabel("PnL", fontsize=10, color="#8b949e")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.3)
    ax1.grid(True, alpha=0.1, color="#30363d")
    ax1.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 2: Drawdown Chart ═══
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    drawdowns = metrics["drawdowns"]
    ax2.fill_between(
        timestamps, 0, drawdowns, color="#f85149", alpha=0.4,
    )
    ax2.plot(
        timestamps, drawdowns, color="#f85149", linewidth=1.2,
    )
    ax2.axhline(y=0, color="#484f58", linewidth=0.5, linestyle="--")
    max_dd = metrics["max_drawdown"]
    if max_dd < 0:
        dd_idx = drawdowns.index(max_dd)
        ax2.annotate(
            f"Max DD: {max_dd:.2f}",
            xy=(timestamps[dd_idx], max_dd),
            xytext=(timestamps[dd_idx], max_dd * 0.6),
            fontsize=9, color="#f85149", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.2),
        )
    ax2.set_title("Drawdown", fontsize=13, color="#c9d1d9", pad=10)
    ax2.set_xlabel("Timestamp", fontsize=10, color="#8b949e")
    ax2.set_ylabel("Drawdown", fontsize=10, color="#8b949e")
    ax2.grid(True, alpha=0.1, color="#30363d")
    ax2.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 3: Mid-Price + Inventory Overlay ═══
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#161b22")

    # Plot mid-prices (left axis)
    plotted_any_mid = False
    for i, product in enumerate(result.products):
        mid_vals = [entry.get(product) for entry in result.mid_price_series]
        ts = [entry["timestamp"] for entry in result.mid_price_series]
        # Filter out None values
        valid = [(t, m) for t, m in zip(ts, mid_vals) if m is not None]
        if valid:
            t_valid, m_valid = zip(*valid)
            short = SHORT_PRODUCT_LABELS.get(product, product)
            color = colors[i % len(colors)]
            ax3.plot(
                t_valid, m_valid, color=color, linewidth=1.0,
                alpha=0.8, label=f"{short} mid",
            )
            plotted_any_mid = True

    ax3.set_ylabel("Mid Price", fontsize=10, color="#8b949e")
    ax3.tick_params(axis="y", colors="#8b949e", labelsize=8)

    # Inventory (right axis)
    ax3_inv = ax3.twinx()
    for i, product in enumerate(result.products):
        inv_vals = [entry.get(product, 0) for entry in result.position_series]
        ts = [entry["timestamp"] for entry in result.position_series]
        short = SHORT_PRODUCT_LABELS.get(product, product)
        color = colors[i % len(colors)]
        ax3_inv.plot(
            ts, inv_vals, color=color, linewidth=0.8,
            alpha=0.5, linestyle="--", label=f"{short} inv",
        )
    ax3_inv.set_ylabel("Inventory", fontsize=10, color="#8b949e")
    ax3_inv.tick_params(axis="y", colors="#8b949e", labelsize=8)

    ax3.set_title(
        "Mid-Price & Inventory Over Time",
        fontsize=13, color="#c9d1d9", pad=10,
    )
    ax3.set_xlabel("Timestamp", fontsize=10, color="#8b949e")
    ax3.grid(True, alpha=0.1, color="#30363d")

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_inv.get_legend_handles_labels()
    ax3.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=7, loc="upper left", framealpha=0.3, ncol=4,
    )
    ax3.tick_params(axis="x", colors="#8b949e", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color("#30363d")
    for spine in ax3_inv.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 4: Return Distribution ═══
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#161b22")
    returns = metrics["returns"]
    if returns:
        ax4.hist(
            returns, bins=min(80, max(20, len(returns) // 20)),
            color="#58a6ff", alpha=0.7, edgecolor="#1f6feb",
        )
        mean_ret = np.mean(returns)
        ax4.axvline(mean_ret, color="#3fb950", linewidth=1.5, linestyle="--",
                     label=f"Mean: {mean_ret:.4f}")
    ax4.set_title("Return Distribution", fontsize=13, color="#c9d1d9", pad=10)
    ax4.set_xlabel("Return (tick-to-tick PnL change)", fontsize=10, color="#8b949e")
    ax4.set_ylabel("Frequency", fontsize=10, color="#8b949e")
    ax4.legend(fontsize=9, framealpha=0.3)
    ax4.grid(True, alpha=0.1, color="#30363d")
    ax4.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax4.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 5: Metrics Summary Table ═══
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#161b22")
    ax5.axis("off")

    table_data = [
        ["Metric", "Value"],
        ["Final PnL", f"{metrics['final_pnl']:.2f}"],
        ["Tick Count", f"{metrics['tick_count']:,}"],
        ["Total Trades", f"{metrics['total_trades']:,}"],
        ["Avg Fill", f"{metrics['avg_fill']:.2f}"],
        ["Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%"],
        ["Profit Factor", f"{metrics['profit_factor']:.4f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']:.2f}"],
        ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}"],
        ["Calmar Ratio", f"{metrics['calmar_ratio']:.4f}"],
        ["Recovery", metrics["recovery"]],
    ]

    table = ax5.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#1f6feb")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0d1117" if row % 2 == 0 else "#161b22")
            cell.set_text_props(color="#c9d1d9")

    ax5.set_title("Performance Metrics", fontsize=13, color="#c9d1d9", pad=10)

    # Always save to file
    if not save_path:
        os.makedirs("runs", exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
        save_path = os.path.join("runs", f"chart_{safe_label}.png")

    _save_and_show(fig, save_path, label)


def visualise_product_comparison(
    results: Dict[str, BacktestResult],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualise product-wise PnL comparison across multiple runs (e.g. Day -1 vs Day -2).
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed — skipping visualisations.")
        return

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Product-wise PnL Comparison",
        fontsize=16, fontweight="bold", color="#58a6ff", y=1.02,
    )

    if len(results) == 1:
        axes = [axes]

    colors = [
        "#58a6ff", "#3fb950", "#f0883e", "#d2a8ff",
        "#ff7b72", "#79c0ff", "#7ee787", "#ffa657",
    ]

    for idx, (run_label, result) in enumerate(results.items()):
        ax = axes[idx]
        ax.set_facecolor("#161b22")
        pm = compute_product_metrics(result)

        products = list(pm.keys())
        short_labels = [SHORT_PRODUCT_LABELS.get(p, p) for p in products]
        pnl_vals = [pm[p]["final_pnl"] for p in products]
        bar_colors = [
            "#3fb950" if v >= 0 else "#f85149" for v in pnl_vals
        ]

        bars = ax.barh(short_labels, pnl_vals, color=bar_colors, alpha=0.8, height=0.6)
        ax.axvline(0, color="#484f58", linewidth=0.5, linestyle="--")
        ax.set_title(run_label, fontsize=13, color="#c9d1d9", pad=10)
        ax.set_xlabel("PnL", fontsize=10, color="#8b949e")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.grid(True, alpha=0.1, color="#30363d", axis="x")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        # Add value labels
        for bar, val in zip(bars, pnl_vals):
            ax.text(
                bar.get_width() + (abs(max(pnl_vals, default=1)) * 0.02 if val >= 0 else -abs(max(pnl_vals, default=1)) * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
                color="#c9d1d9",
            )

    plt.tight_layout()

    # Always save to file
    if not save_path:
        os.makedirs("runs", exist_ok=True)
        save_path = os.path.join("runs", "chart_product_comparison.png")

    _save_and_show(fig, save_path, "comparison")


# ─── Fair Value + Trade Markers ──────────────────────────────────────────────


def visualise_fair_value_trades(
    result: BacktestResult,
    label: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Per-product fair-value timeseries with executed trade markers.

    For each product:
      • Line: forward-rolling-regression fair value
      • ▲ green triangle  = buy trade (at trade price)
      • ▼ red   triangle  = sell trade (at trade price)
      • ✓ green above/below = trade was profitable (pnl_contribution > 0)
      • ✗ red   above/below = trade was a loss     (pnl_contribution ≤ 0)
      • Win rate annotation for the day
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed — skipping visualisations.")
        return

    fair_value_series = getattr(result, "fair_value_series", [])
    own_trades_enriched = getattr(result, "own_trades_enriched", [])

    if not fair_value_series:
        print("[WARN] No fair value data — skipping fair value visualisation.")
        return

    products = result.products
    n = len(products)
    if n == 0:
        return

    plt.style.use("dark_background")
    fig, axes = plt.subplots(n, 1, figsize=(20, 5 * n), squeeze=False)
    fig.patch.set_facecolor("#0d1117")

    title = f"Fair Value & Trade Analysis{f'  —  {label}' if label else ''}"
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#58a6ff", y=0.99)

    fv_timestamps = [e["timestamp"] for e in fair_value_series]

    colors = [
        "#58a6ff", "#3fb950", "#f0883e", "#d2a8ff",
        "#ff7b72", "#79c0ff", "#7ee787", "#ffa657",
    ]

    for row_idx, product in enumerate(products):
        ax = axes[row_idx][0]
        ax.set_facecolor("#161b22")

        # ── Fair value timeseries ──
        fv_vals = [e.get(product) for e in fair_value_series]
        valid_fv = [(t, v) for t, v in zip(fv_timestamps, fv_vals) if v is not None]
        fv_range = 1.0  # fallback for offset calculations

        if valid_fv:
            ts_v, fv_v_list = zip(*valid_fv)
            fv_v = list(fv_v_list)
            fv_color = colors[row_idx % len(colors)]
            ax.plot(ts_v, fv_v, color=fv_color, linewidth=1.3,
                    label="Fair Value (regression)", zorder=3, alpha=0.9)
            fv_range = max(max(fv_v) - min(fv_v), 1.0)

        # Also plot CSV mid price faintly for reference
        mid_vals = [e.get(product) for e in result.mid_price_series]
        valid_mid = [(t, v) for t, v in zip(
            [e["timestamp"] for e in result.mid_price_series], mid_vals
        ) if v is not None]
        if valid_mid:
            ts_m, mp_v = zip(*valid_mid)
            ax.plot(ts_m, mp_v, color="#484f58", linewidth=0.7,
                    linestyle=":", label="Mid Price", zorder=2, alpha=0.6)

        # ── Trade markers ──
        product_trades = [t for t in own_trades_enriched if t["symbol"] == product]
        y_offset = fv_range * 0.025  # small vertical nudge for annotations

        if product_trades:
            buys  = [t for t in product_trades if t["is_buy"]]
            sells = [t for t in product_trades if not t["is_buy"]]

            # Buys — green up-triangles
            if buys:
                buy_ts     = [t["timestamp"] for t in buys]
                buy_prices = [t["price"] for t in buys]
                ax.scatter(buy_ts, buy_prices, marker="^", color="#3fb950",
                           s=90, zorder=5, label="Buy", alpha=0.95, edgecolors="#ffffff",
                           linewidths=0.4)
                # Win/loss symbol above each buy marker
                for t in buys:
                    sym   = "✓" if t["pnl_contribution"] > 0 else "✗"
                    color = "#3fb950" if t["pnl_contribution"] > 0 else "#f85149"
                    ax.text(
                        t["timestamp"], t["price"] + y_offset * 1.6,
                        sym, ha="center", va="bottom",
                        fontsize=7, color=color, fontweight="bold", zorder=6,
                    )

            # Sells — red down-triangles
            if sells:
                sell_ts     = [t["timestamp"] for t in sells]
                sell_prices = [t["price"] for t in sells]
                ax.scatter(sell_ts, sell_prices, marker="v", color="#f85149",
                           s=90, zorder=5, label="Sell", alpha=0.95, edgecolors="#ffffff",
                           linewidths=0.4)
                # Win/loss symbol below each sell marker
                for t in sells:
                    sym   = "✓" if t["pnl_contribution"] > 0 else "✗"
                    color = "#3fb950" if t["pnl_contribution"] > 0 else "#f85149"
                    ax.text(
                        t["timestamp"], t["price"] - y_offset * 1.6,
                        sym, ha="center", va="top",
                        fontsize=7, color=color, fontweight="bold", zorder=6,
                    )

            # Win rate annotation
            total = len(product_trades)
            wins  = sum(1 for t in product_trades if t["pnl_contribution"] > 0)
            wr    = wins / total * 100 if total > 0 else 0.0
            wr_color = "#3fb950" if wr >= 50 else "#f85149"
            ax.text(
                0.01, 0.97,
                f"Win Rate: {wr:.1f}%  ({wins}/{total} trades)",
                transform=ax.transAxes,
                fontsize=10, color=wr_color, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                          edgecolor="#30363d", alpha=0.85),
                zorder=7,
            )
        else:
            ax.text(
                0.01, 0.97, "No trades",
                transform=ax.transAxes, fontsize=9, color="#8b949e",
                va="top", ha="left",
            )

        short = SHORT_PRODUCT_LABELS.get(product, product)
        ax.set_title(f"{short}  —  Fair Value & Trades", fontsize=12,
                     color="#c9d1d9", pad=8)
        ax.set_xlabel("Timestamp", fontsize=9, color="#8b949e")
        ax.set_ylabel("Price", fontsize=9, color="#8b949e")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.3, ncol=4)
        ax.grid(True, alpha=0.1, color="#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if not save_path:
        os.makedirs("runs", exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
        save_path = os.path.join("runs", f"chart_{safe_label}_fair_value.png")

    _save_and_show(fig, save_path, label)


# ─── Loss-Point Order Book Analysis (L2/L3) ──────────────────────────────────


def visualise_loss_book_analysis(
    result: BacktestResult,
    label: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    For each product, examine the order book at loss timestamps:

    • "Flat L2 price" — level-2 bid price == level-1 bid price, OR
                         level-2 ask price == level-1 ask price
                         (same price repeated across book levels)
    • "Non-empty L3"  — level-3 bid OR ask has non-zero volume

    Displays percentage of loss trades that hit each condition as a grouped bar chart.
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed — skipping visualisations.")
        return

    own_trades_enriched = getattr(result, "own_trades_enriched", [])
    book_snapshot_series = getattr(result, "book_snapshot_series", [])

    if not own_trades_enriched or not book_snapshot_series:
        print("[WARN] No enriched trade / book snapshot data — skipping L2/L3 analysis.")
        return

    # Build fast timestamp → book_snapshot lookup
    book_by_ts: Dict[int, Any] = {e["timestamp"]: e for e in book_snapshot_series}

    analysis: Dict[str, Dict[str, Any]] = {}
    for product in result.products:
        loss_trades = [
            t for t in own_trades_enriched
            if t["symbol"] == product and t["pnl_contribution"] < 0
        ]
        if not loss_trades:
            continue

        total_losses = len(loss_trades)
        flat_l2_count   = 0
        nonempty_l3_count = 0

        for trade in loss_trades:
            snap = book_by_ts.get(trade["timestamp"])
            if snap is None:
                continue
            prod_book = snap.get(product)
            if prod_book is None:
                continue

            bids = prod_book.get("bids", [])   # list of (price, vol) tuples, sorted desc
            asks = prod_book.get("asks", [])   # sorted asc

            # Flat L2: L2 bid price == L1 bid price  OR  L2 ask price == L1 ask price
            flat_l2 = False
            if len(bids) >= 2 and bids[1][0] == bids[0][0]:
                flat_l2 = True
            if len(asks) >= 2 and asks[1][0] == asks[0][0]:
                flat_l2 = True
            if flat_l2:
                flat_l2_count += 1

            # Non-empty L3: L3 bid or ask has positive volume
            nonempty_l3 = False
            if len(bids) >= 3 and bids[2][1] > 0:
                nonempty_l3 = True
            if len(asks) >= 3 and asks[2][1] > 0:
                nonempty_l3 = True
            if nonempty_l3:
                nonempty_l3_count += 1

        analysis[product] = {
            "total_losses": total_losses,
            "flat_l2_count": flat_l2_count,
            "flat_l2_pct": flat_l2_count / total_losses * 100,
            "nonempty_l3_count": nonempty_l3_count,
            "nonempty_l3_pct": nonempty_l3_count / total_losses * 100,
        }

    # Print summary
    print(f"\n─── Loss-Point Book Analysis{f'  —  {label}' if label else ''} ───")
    for product, stats in analysis.items():
        short = SHORT_PRODUCT_LABELS.get(product, product)
        print(
            f"  {short:<12s}  Losses: {stats['total_losses']:>4d}  "
            f"Flat-L2: {stats['flat_l2_pct']:>5.1f}%  "
            f"Non-empty-L3: {stats['nonempty_l3_pct']:>5.1f}%"
        )

    if not analysis:
        print("[INFO] No loss trades found — skipping L2/L3 chart.")
        return

    plt.style.use("dark_background")
    products_with_losses = list(analysis.keys())
    n_prod = len(products_with_losses)
    fig_w = max(10, n_prod * 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    title = f"Loss-Point Book Structure (L2 / L3){f'  —  {label}' if label else ''}"
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#58a6ff")

    short_labels = [SHORT_PRODUCT_LABELS.get(p, p) for p in products_with_losses]
    x = np.arange(n_prod)
    width = 0.35

    flat_l2_pcts     = [analysis[p]["flat_l2_pct"]     for p in products_with_losses]
    nonempty_l3_pcts = [analysis[p]["nonempty_l3_pct"] for p in products_with_losses]

    bars1 = ax.bar(x - width / 2, flat_l2_pcts,     width,
                   label="Flat L2 price (%)", color="#f0883e", alpha=0.85)
    bars2 = ax.bar(x + width / 2, nonempty_l3_pcts, width,
                   label="Non-empty L3 bid/ask (%)", color="#d2a8ff", alpha=0.85)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8, color="#c9d1d9")
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8, color="#c9d1d9")

    # Loss count annotations below x-axis
    for i, p in enumerate(products_with_losses):
        ax.text(x[i], -4,
                f"n={analysis[p]['total_losses']} losses",
                ha="center", va="top", fontsize=8, color="#8b949e")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10, color="#c9d1d9")
    ax.set_ylabel("% of Loss Trades", fontsize=10, color="#8b949e")
    ax.set_ylim(-12, 110)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.1, color="#30363d", axis="y")
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    # Explanatory footnote
    fig.text(
        0.01, 0.01,
        "Flat L2: level-2 bid or ask price equals level-1 price  |  "
        "Non-empty L3: level-3 bid or ask has non-zero volume",
        fontsize=7, color="#484f58", va="bottom",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if not save_path:
        os.makedirs("runs", exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
        save_path = os.path.join("runs", f"chart_{safe_label}_loss_book.png")

    _save_and_show(fig, save_path, label)





def full_analysis(
    result: BacktestResult,
    label: str = "",
    save_prefix: Optional[str] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """Run full analysis: print metrics + generate all charts."""
    metrics = compute_metrics(result)
    product_metrics = compute_product_metrics(result)

    print_metrics(metrics, label)
    print_product_table(product_metrics, label)

    safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
    if save_prefix:
        save_path = f"{save_prefix}_{safe_label}.png"
    else:
        os.makedirs("runs", exist_ok=True)
        save_path = os.path.join("runs", f"chart_{safe_label}.png")

    visualise(result, label=label, save_path=save_path, show=show)

    return metrics


# ─── Log Saving ──────────────────────────────────────────────────────────────


def save_run_log(result: BacktestResult, label: str = "") -> str:
    """Save full backtest log (metrics + PnL series + trades) to a JSON file."""
    os.makedirs("runs", exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
    log_path = os.path.join("runs", f"log_{safe_label}.json")

    metrics = compute_metrics(result)
    product_metrics = compute_product_metrics(result)

    log_data = {
        "label": label,
        "final_pnl": metrics["final_pnl"],
        "tick_count": metrics["tick_count"],
        "total_trades": metrics["total_trades"],
        "avg_fill": metrics["avg_fill"],
        "max_drawdown": metrics["max_drawdown"],
        "profit_factor": metrics["profit_factor"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "calmar_ratio": metrics["calmar_ratio"],
        "recovery": metrics["recovery"],
        "pnl_by_product": {
            k: v for k, v in result.metrics.final_pnl_by_product.items()
        },
        "product_metrics": product_metrics,
        "pnl_series": result.pnl_series,
        "position_series": result.position_series,
        "own_trades": [
            {
                "symbol": t.symbol,
                "price": t.price,
                "quantity": t.quantity,
                "buyer": t.buyer,
                "seller": t.seller,
                "timestamp": t.timestamp,
            }
            for t in result.own_trades_all
        ],
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"  [LOG] {log_path}")
    return log_path
