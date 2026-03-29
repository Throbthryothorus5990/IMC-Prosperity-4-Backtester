# Python IMC Prosperity 4 Backtester

Pure-Python port of the [Rust IMC Prosperity 4 backtester](https://github.com/GeyzsoN/prosperity_rust_backtester).

Same matching logic, same PnL calculations, same position limits — now in Python with built-in visualisations.

## Features

- **Exact matching engine** — order matching against book + market trades with queue penetration, slippage, and position limits
- **Same PnL tracking** — cash + mark-to-market per product
- **Built-in visualiser** — PnL charts, drawdown analysis, mid-price/inventory overlay, return distributions
- **Full metrics** — Sharpe, Calmar, max drawdown, recovery, and more
- **Google Colab ready** — clone, overwrite trader, run

## Quick Start

### Local

```bash
# 1. Clone
git clone "https://github.com/Akgithub2028/IMC-Prosperity-4-Backtester"
cd prosperity_backtester

# 2. Install deps
pip install -r requirements.txt

# 3. Download datasets
python colab_setup.py

# 4. Run with default trader
python run_backtest.py

# 5. Run with your own trader
python run_backtest.py --trader my_strategy.py
```

### Google Colab

```python
# Cell 1: Clone and setup
!git clone "https://github.com/Akgithub2028/IMC-Prosperity-4-Backtester"
%cd IMC-Prosperity-4-Backtester
!pip install -q matplotlib numpy
!python colab_setup.py

# Cell 2: Write your trader
%%writefile solution.py
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

class Trader:
    def run(self, state: TradingState):
        orders = {}
        for product, depth in state.order_depths.items():
            orders[product] = []
        return orders, 0, ""

# Cell 3: Run backtest
%run run_backtest.py
```

## CLI Options

```bash
python run_backtest.py --help

# Run specific day
python run_backtest.py --day -1

# Specify dataset
python run_backtest.py --dataset datasets/tutorial

# Matching options
python run_backtest.py --trade-match-mode worse --queue-penetration 0.5

# Save charts without display
python run_backtest.py --save-charts output --no-visualise
```

## Directory Structure

```
prosperity_backtester/
├── datamodel.py       # IMC data model (Listing, Order, Trade, TradingState, etc.)
├── backtester.py      # Core engine: dataset loading + matching + PnL
├── runner.py          # CLI runner + summary printing
├── visualiser.py      # Charts + metrics
├── solution.py        # Your trader class (overwrite this!)
├── run_backtest.py    # Main entry point
├── colab_setup.py     # Dataset downloader
├── requirements.txt
├── datasets/
│   └── tutorial/      # Bundled tutorial data
│       ├── prices_round_0_day_-1.csv
│       ├── trades_round_0_day_-1.csv
│       ├── prices_round_0_day_-2.csv
│       ├── trades_round_0_day_-2.csv
│       └── submission.log
└── runs/              # Output directory
```

## Metrics

| Metric | Description |
|--------|-------------|
| Max % Drawdown | Largest peak-to-trough percentage drop |
| Max Drawdown | Largest absolute drawdown |
| Avg % Drawdown | Mean of all drawdown segments |
| Sharpe Ratio | Annualized return / volatility |
| Calmar Ratio | Total return / max drawdown |
| Total Trades | Count of own trades |
| Avg Fill | Average trade fill quantity |
| Recovery | Ticks from trough to new high |
|Profit Factor|	Gross profit divided by gross loss. (A value > 1.0 means profitable) |
## Position Limits (per product)

These are hardcoded to match the Rust backtester and IMC competition rules:

| Product | Limit |
|---------|-------|
| EMERALDS | 80 |
| TOMATOES | 80 |
| RAINFOREST_RESIN | 50 |
| KELP | 50 |
| SQUID_INK | 50 |
| CROISSANTS | 250 |
| JAMS | 350 |
| DJEMBES | 60 |
| PICNIC_BASKET1 | 60 |
| PICNIC_BASKET2 | 100 |
| VOLCANIC_ROCK | 400 |
| VOLCANIC_ROCK_VOUCHER_* | 200 |
| MAGNIFICENT_MACARONS | 75 |
| Others | 100 |

## License

MIT / Apache-2.0 (same as original Rust backtester)
