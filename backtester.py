"""
Core backtester engine — faithful Python port of model.rs + runner.rs from
the Rust IMC Prosperity 4 backtester.

Every matching-engine detail (position limits, queue penetration with banker's
rounding, slippage, trade-match modes, mark-to-market PnL) is preserved exactly.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import importlib.util
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datamodel import (
    Listing,
    Observation,
    ConversionObservation,
    Order as DmOrder,
    OrderDepth,
    Trade as DmTrade,
    TradingState,
)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_POSITION_LIMIT = 100
LOG_CHAR_LIMIT = 3750
ACTIVITY_HEADER = (
    "day;timestamp;product;"
    "bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;"
    "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;"
    "mid_price;profit_and_loss"
)
ACTIVITY_HEADER_PREFIX = (
    "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2"
)
TRADE_HEADER_PREFIX = "timestamp;buyer;seller;symbol;currency;price;quantity"

POSITION_LIMITS: Dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}

SHORT_PRODUCT_LABELS: Dict[str, str] = {
    "EMERALDS": "EMR",
    "TOMATOES": "TOM",
    "RAINFOREST_RESIN": "RESIN",
    "KELP": "KELP",
    "SQUID_INK": "SQUID",
    "CROISSANTS": "CROISS",
    "JAMS": "JAMS",
    "DJEMBES": "DJEMBE",
    "PICNIC_BASKET1": "PB1",
    "PICNIC_BASKET2": "PB2",
    "VOLCANIC_ROCK": "ROCK",
    "VOLCANIC_ROCK_VOUCHER_9500": "V9500",
    "VOLCANIC_ROCK_VOUCHER_9750": "V9750",
    "VOLCANIC_ROCK_VOUCHER_10000": "V10000",
    "VOLCANIC_ROCK_VOUCHER_10250": "V10250",
    "VOLCANIC_ROCK_VOUCHER_10500": "V10500",
    "MAGNIFICENT_MACARONS": "MACARON",
}

# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class OrderBookLevel:
    price: int
    volume: int


@dataclass
class MarketTrade:
    symbol: str
    price: int
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0


@dataclass
class ProductSnapshot:
    product: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    mid_price: Optional[float] = None


@dataclass
class ObservationState:
    plain: Dict[str, int] = field(default_factory=dict)
    conversion: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TickSnapshot:
    timestamp: int
    day: Optional[int] = None
    products: "OrderedDict[str, ProductSnapshot]" = field(
        default_factory=OrderedDict
    )
    market_trades: "Dict[str, List[MarketTrade]]" = field(default_factory=dict)
    observations: ObservationState = field(default_factory=ObservationState)


@dataclass
class NormalizedDataset:
    schema_version: str = "1.0"
    competition_version: str = "p4"
    dataset_id: str = ""
    source: str = ""
    products: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ticks: List[TickSnapshot] = field(default_factory=list)


@dataclass
class MatchingConfig:
    trade_match_mode: str = "all"
    queue_penetration: float = 1.0
    price_slippage_bps: float = 0.0


@dataclass
class InternalOrder:
    symbol: str
    price: int
    quantity: int


@dataclass
class InternalTrade:
    symbol: str
    price: int
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0


@dataclass
class RunRequest:
    trader_file: str
    dataset_file: str
    day: Optional[int] = None
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    run_id: Optional[str] = None
    output_root: str = "runs"


@dataclass
class RunMetrics:
    run_id: str = ""
    dataset_id: str = ""
    dataset_path: str = ""
    trader_path: str = ""
    day: Optional[int] = None
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    tick_count: int = 0
    own_trade_count: int = 0
    final_pnl_total: float = 0.0
    final_pnl_by_product: "OrderedDict[str, float]" = field(
        default_factory=OrderedDict
    )
    generated_at: str = ""


@dataclass
class BacktestResult:
    """All data produced by a single backtest run."""

    metrics: RunMetrics = field(default_factory=RunMetrics)
    pnl_series: List[Dict[str, Any]] = field(default_factory=list)
    position_series: List[Dict[str, Any]] = field(default_factory=list)
    mid_price_series: List[Dict[str, Any]] = field(default_factory=list)
    fair_value_series: List[Dict[str, Any]] = field(default_factory=list)
    market_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    own_trades_all: List[InternalTrade] = field(default_factory=list)
    products: List[str] = field(default_factory=list)


# ─── Python's banker's rounding (identical to Rust round_ties_even) ───────────


def _python_round(value: float) -> int:
    """Round half-to-even, matching Rust f64::round_ties_even()."""
    return round(value)


def _python_round_digits(value: float, digits: int) -> float:
    return round(value, digits)


# ─── Dataset Loading ──────────────────────────────────────────────────────────


def load_dataset(path: str) -> NormalizedDataset:
    """Load a dataset from CSV, JSON, or submission log."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_price_csv_dataset(path)
    elif ext == ".log":
        return _load_submission_log_dataset(path)
    elif ext == ".json":
        return _load_json_dataset(path)
    else:
        raise ValueError(
            f"Unsupported dataset format for {path}; "
            "expected JSON, prices CSV, or submission log"
        )


def _load_json_dataset(path: str) -> NormalizedDataset:
    with open(path, "r") as f:
        payload = f.read()

    data = json.loads(payload)

    # Try as NormalizedDataset first
    if "ticks" in data and "products" in data:
        return _json_to_normalized(data)

    # Try as submission payload
    if "activitiesLog" in data and isinstance(data["activitiesLog"], str):
        return _load_submission_value_dataset(path, data)

    raise ValueError(f"Cannot parse supported dataset JSON: {path}")


def _json_to_normalized(data: dict) -> NormalizedDataset:
    ds = NormalizedDataset(
        schema_version=data.get("schema_version", "1.0"),
        competition_version=data.get("competition_version", "p4"),
        dataset_id=data.get("dataset_id", ""),
        source=data.get("source", ""),
        products=data.get("products", []),
        metadata=data.get("metadata", {}),
    )
    for tick_data in data.get("ticks", []):
        tick = TickSnapshot(
            timestamp=tick_data["timestamp"],
            day=tick_data.get("day"),
        )
        for prod_name, prod_data in tick_data.get("products", {}).items():
            snap = ProductSnapshot(
                product=prod_data.get("product", prod_name),
                mid_price=prod_data.get("mid_price"),
            )
            for bid_data in prod_data.get("bids", []):
                snap.bids.append(
                    OrderBookLevel(int(bid_data["price"]), int(bid_data["volume"]))
                )
            for ask_data in prod_data.get("asks", []):
                snap.asks.append(
                    OrderBookLevel(int(ask_data["price"]), int(ask_data["volume"]))
                )
            tick.products[prod_name] = snap

        for symbol, trades_data in tick_data.get("market_trades", {}).items():
            tick.market_trades[symbol] = [
                MarketTrade(
                    symbol=t.get("symbol", symbol),
                    price=int(t["price"]),
                    quantity=int(t["quantity"]),
                    buyer=t.get("buyer", ""),
                    seller=t.get("seller", ""),
                    timestamp=int(t.get("timestamp", 0)),
                )
                for t in trades_data
            ]

        obs_data = tick_data.get("observations", {})
        tick.observations.plain = {
            k: int(v) for k, v in obs_data.get("plain", {}).items()
        }
        tick.observations.conversion = obs_data.get("conversion", {})
        ds.ticks.append(tick)
    return ds


def _load_submission_log_dataset(path: str) -> NormalizedDataset:
    with open(path, "r") as f:
        payload = f.read()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        raise ValueError(f"Cannot parse submission log JSON: {path}")

    if "activitiesLog" not in value or not isinstance(value["activitiesLog"], str):
        raise ValueError(f"Not a valid submission log: {path}")

    return _load_submission_value_dataset(path, value)


def _load_submission_value_dataset(
    path: str, value: dict
) -> NormalizedDataset:
    activities_log: str = value["activitiesLog"]
    trade_history: List[MarketTrade] = []

    if "tradeHistory" in value and isinstance(value["tradeHistory"], list):
        for row in value["tradeHistory"]:
            trade_history.append(
                MarketTrade(
                    symbol=row.get("symbol", ""),
                    price=_parse_trade_value_price(row.get("price")),
                    quantity=int(row.get("quantity", 0)),
                    buyer=row.get("buyer", ""),
                    seller=row.get("seller", ""),
                    timestamp=int(row.get("timestamp", 0)),
                )
            )

    metadata = {"built_from": path}
    did = _submission_dataset_id_from_path(path)
    return _build_dataset_from_activities(
        path, did, path, activities_log, trade_history, metadata
    )


def _load_price_csv_dataset(path: str) -> NormalizedDataset:
    base = os.path.basename(path)
    if not base.startswith("prices_"):
        raise ValueError(
            f"Unsupported CSV input {path}; pass a prices_*.csv file"
        )

    with open(path, "r") as f:
        activities_log = f.read()

    trade_history: List[MarketTrade] = []
    trades_path = path.replace("prices_", "trades_", 1)
    if os.path.isfile(trades_path):
        trade_history = _load_trades_csv(trades_path)

    metadata = {
        "source_format": "imc_csv",
        "trade_rows": len(trade_history),
    }
    did = os.path.splitext(os.path.basename(path))[0]
    source = f"imc_csv:{base}"
    return _build_dataset_from_activities(
        path, did, source, activities_log, trade_history, metadata
    )


def _load_trades_csv(path: str) -> List[MarketTrade]:
    with open(path, "r") as f:
        payload = f.read()
    trades: List[MarketTrade] = []
    for line_number, line in enumerate(payload.splitlines()):
        if line_number == 0:
            if not line.startswith(TRADE_HEADER_PREFIX):
                raise ValueError(f"Unexpected trades header in {path}")
            continue
        if not line.strip():
            continue
        fields = line.split(";")
        if len(fields) < 7:
            raise ValueError(
                f"Invalid trades row {line_number + 1} in {path}; expected 7 columns"
            )
        trades.append(
            MarketTrade(
                timestamp=int(fields[0].strip()),
                buyer=fields[1].strip(),
                seller=fields[2].strip(),
                symbol=fields[3].strip(),
                price=_parse_price_i64(fields[5]),
                quantity=int(fields[6].strip()),
            )
        )
    return trades


def _build_dataset_from_activities(
    path: str,
    dataset_id: str,
    source: str,
    activities_log: str,
    trade_history: List[MarketTrade],
    metadata: Dict[str, Any],
) -> NormalizedDataset:
    products_seen: OrderedDict[str, None] = OrderedDict()
    ticks_by_key: Dict[Tuple[Optional[int], int], TickSnapshot] = {}
    activity_row_count = 0

    for line_number, line in enumerate(activities_log.splitlines()):
        if line_number == 0:
            if not line.startswith(ACTIVITY_HEADER_PREFIX):
                raise ValueError(f"Unexpected activities header in {path}")
            continue
        if not line.strip():
            continue

        fields = line.split(";")
        if len(fields) < 17:
            raise ValueError(
                f"Invalid activities row {line_number + 1} in {path}; "
                "expected at least 17 columns"
            )

        day = _parse_optional_int(fields[0])
        timestamp = int(fields[1].strip())
        product = fields[2].strip()
        if not product:
            raise ValueError(
                f"Missing product in activities row {line_number + 1} of {path}"
            )

        snapshot = ProductSnapshot(
            product=product,
            bids=_parse_book_side(fields, [(3, 4), (5, 6), (7, 8)]),
            asks=_parse_book_side(fields, [(9, 10), (11, 12), (13, 14)]),
            mid_price=_parse_optional_float(fields[15]),
        )

        activity_row_count += 1
        products_seen[product] = None

        key = (day, timestamp)
        if key not in ticks_by_key:
            ticks_by_key[key] = TickSnapshot(
                timestamp=timestamp,
                day=day,
                products=OrderedDict(),
                market_trades={},
                observations=ObservationState(),
            )
        ticks_by_key[key].products[product] = snapshot

    if not ticks_by_key:
        raise ValueError(f"No tick rows found in {path}")

    # Group trades by timestamp
    trades_by_ts: Dict[int, Dict[str, List[MarketTrade]]] = {}
    for trade in trade_history:
        ts_trades = trades_by_ts.setdefault(trade.timestamp, {})
        ts_trades.setdefault(trade.symbol, []).append(trade)

    # Sort ticks by (day, timestamp)
    sorted_keys = sorted(ticks_by_key.keys())
    ticks: List[TickSnapshot] = []
    for key in sorted_keys:
        tick = ticks_by_key[key]
        if tick.timestamp in trades_by_ts:
            tick.market_trades = trades_by_ts.pop(tick.timestamp)
        ticks.append(tick)

    products = sorted(products_seen.keys())
    full_metadata = {"activity_rows": activity_row_count}
    full_metadata.update(metadata)
    if "trade_rows" not in full_metadata:
        full_metadata["trade_rows"] = len(trade_history)

    return NormalizedDataset(
        schema_version="1.0",
        competition_version="p4",
        dataset_id=dataset_id,
        source=source,
        products=products,
        metadata=full_metadata,
        ticks=ticks,
    )


def _parse_book_side(
    fields: List[str], pairs: List[Tuple[int, int]]
) -> List[OrderBookLevel]:
    levels = []
    for price_idx, volume_idx in pairs:
        if price_idx >= len(fields) or volume_idx >= len(fields):
            continue
        pt = fields[price_idx].strip()
        vt = fields[volume_idx].strip()
        if not pt or not vt:
            continue
        levels.append(
            OrderBookLevel(
                price=_parse_price_i64(pt),
                volume=int(vt),
            )
        )
    return levels


def _parse_optional_int(value: str) -> Optional[int]:
    v = value.strip()
    if not v:
        return None
    return int(v)


def _parse_optional_float(value: str) -> Optional[float]:
    v = value.strip()
    if not v:
        return None
    return float(v)


def _parse_price_i64(value: str) -> int:
    return _python_round(float(value.strip()))


def _parse_trade_value_price(value) -> int:
    if value is None:
        raise ValueError("tradeHistory row missing price")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _python_round(value)
    if isinstance(value, str):
        return _parse_price_i64(value)
    raise ValueError("tradeHistory row has unsupported price value")


def _submission_dataset_id_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem and stem.isdigit():
        return f"official_submission_{stem}_alltrades"
    return stem


# ─── Trader Loading ───────────────────────────────────────────────────────────


def load_trader(trader_file: str):
    """
    Load a Trader class from a Python file, exactly like the Rust backtester does
    via PyO3 (pytrader.rs load_trader_instance).
    """
    spec = importlib.util.spec_from_file_location("user_trader", trader_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load trader file: {trader_file}")

    # Add trader directory and project root to sys.path
    trader_dir = os.path.dirname(os.path.abspath(trader_file))
    project_dir = os.path.dirname(os.path.abspath(__file__))
    for p in [trader_dir, project_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Ensure datamodel is importable — load it from our project directory
    dm_path = os.path.join(project_dir, "datamodel.py")
    dm_spec = importlib.util.spec_from_file_location("datamodel", dm_path)
    if dm_spec and dm_spec.loader:
        dm_module = importlib.util.module_from_spec(dm_spec)
        dm_spec.loader.exec_module(dm_module)
    else:
        raise RuntimeError(f"Cannot load datamodel from {dm_path}")

    # Register datamodel in sys.modules so `from datamodel import ...` works
    sys.modules["datamodel"] = dm_module

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Trader"):
        raise RuntimeError("Trader file does not define a Trader class")

    return module.Trader()


# ─── Order Matching Engine (port of runner.rs) ────────────────────────────────


def position_limit(symbol: str) -> int:
    return POSITION_LIMITS.get(symbol, DEFAULT_POSITION_LIMIT)


def _market_trade_duplicates_touch(
    trade: MarketTrade,
    best_bid: Optional[int],
    best_ask: Optional[int],
) -> bool:
    """Filters out market trades that duplicate the book touch."""
    if trade.buyer == "SUBMISSION":
        if best_ask is not None and trade.price >= best_ask:
            return True
    if trade.seller == "SUBMISSION":
        if best_bid is not None and trade.price <= best_bid:
            return True
    return False


def _queue_penetration_available(quantity: int, queue_penetration: float) -> int:
    raw = quantity * max(0.0, queue_penetration)
    available = _python_round(raw)
    if quantity > 0 and queue_penetration > 0.0 and available == 0:
        available = 1
    return max(0, available)


def _eligible_trade_price(
    order_price: int, trade_price: int, quantity: int, mode: str
) -> bool:
    if mode == "none":
        return False
    if quantity > 0:
        if mode == "all":
            return trade_price <= order_price
        return trade_price < order_price
    if quantity < 0:
        if mode == "all":
            return trade_price >= order_price
        return trade_price > order_price
    return False


def _slippage_adjusted_price(price: int, is_buy: bool, bps: float) -> int:
    if bps <= 0.0:
        return price
    factor = 1.0 + (bps / 10_000.0)
    if is_buy:
        adjusted = price * factor
    else:
        adjusted = price / factor
    return _python_round(adjusted)


def _enforce_position_limits(
    position: Dict[str, int],
    orders_by_symbol: Dict[str, List[InternalOrder]],
) -> Tuple[Dict[str, List[InternalOrder]], List[str]]:
    filtered: Dict[str, List[InternalOrder]] = {}
    messages: List[str] = []

    for symbol, orders in orders_by_symbol.items():
        product_position = position.get(symbol, 0)
        total_long = sum(max(0, o.quantity) for o in orders)
        total_short = sum(max(0, -o.quantity) for o in orders)
        limit = position_limit(symbol)

        if (
            product_position + total_long > limit
            or product_position - total_short < -limit
        ):
            messages.append(
                f"Orders for product {symbol} exceeded limit {limit}; "
                f"product orders canceled for this tick"
            )
            continue

        filtered[symbol] = orders

    return filtered, messages


@dataclass
class _BookLevel:
    price: int
    volume: int


def _match_orders_for_symbol(
    symbol: str,
    orders: List[InternalOrder],
    bids: List[_BookLevel],
    asks: List[_BookLevel],
    market_trades: List[MarketTrade],
    position: Dict[str, int],
    cash_by_product: Dict[str, float],
    timestamp: int,
    config: MatchingConfig,
) -> Tuple[List[InternalTrade], List[InternalTrade]]:
    """
    Match orders against the order book and market trades.
    Returns (own_trades, remaining_market_trades).
    """
    own_trades: List[InternalTrade] = []

    best_bid = max((l.price for l in bids if l.volume > 0), default=None)
    best_ask = min((l.price for l in asks if l.volume > 0), default=None)

    # Build available market trades, filtering duplicates and applying queue penetration
    market_available: List[InternalTrade] = []
    for trade in market_trades:
        if _market_trade_duplicates_touch(trade, best_bid, best_ask):
            continue
        market_available.append(
            InternalTrade(
                symbol=symbol,
                price=trade.price,
                quantity=_queue_penetration_available(
                    trade.quantity, config.queue_penetration
                ),
                buyer=trade.buyer,
                seller=trade.seller,
                timestamp=trade.timestamp if trade.timestamp != 0 else timestamp,
            )
        )

    # Build queue-ahead tracking
    buy_queue_remaining: Dict[int, int] = {
        l.price: l.volume for l in bids if l.volume > 0
    }
    sell_queue_remaining: Dict[int, int] = {
        l.price: l.volume for l in asks if l.volume > 0
    }

    for order in orders:
        remaining = order.quantity

        if remaining > 0:
            # Buy order — match against asks
            for level in asks:
                if remaining <= 0:
                    break
                if level.price > order.price or level.volume <= 0:
                    continue
                fill = min(remaining, level.volume)
                trade_price = _slippage_adjusted_price(
                    level.price, True, config.price_slippage_bps
                )
                own_trades.append(
                    InternalTrade(
                        symbol=symbol,
                        price=trade_price,
                        quantity=fill,
                        buyer="SUBMISSION",
                        seller="",
                        timestamp=timestamp,
                    )
                )
                position[symbol] = position.get(symbol, 0) + fill
                cash_by_product[symbol] = cash_by_product.get(
                    symbol, 0.0
                ) - (trade_price * fill)
                level.volume -= fill
                remaining -= fill

        elif remaining < 0:
            # Sell order — match against bids
            for level in bids:
                if remaining >= 0:
                    break
                if level.price < order.price or level.volume <= 0:
                    continue
                fill = min(-remaining, level.volume)
                trade_price = _slippage_adjusted_price(
                    level.price, False, config.price_slippage_bps
                )
                own_trades.append(
                    InternalTrade(
                        symbol=symbol,
                        price=trade_price,
                        quantity=fill,
                        buyer="",
                        seller="SUBMISSION",
                        timestamp=timestamp,
                    )
                )
                position[symbol] = position.get(symbol, 0) - fill
                cash_by_product[symbol] = cash_by_product.get(
                    symbol, 0.0
                ) + (trade_price * fill)
                level.volume -= fill
                remaining += fill

        # Match remaining against market trades
        if remaining != 0 and config.trade_match_mode != "none":
            for mt in market_available:
                if remaining == 0:
                    break
                if mt.quantity <= 0:
                    continue
                if not _eligible_trade_price(
                    order.price, mt.price, remaining, config.trade_match_mode
                ):
                    continue

                # Queue-ahead deduction
                if remaining > 0 and mt.price == order.price:
                    if order.price in buy_queue_remaining:
                        consumed = min(mt.quantity, buy_queue_remaining[order.price])
                        mt.quantity -= consumed
                        buy_queue_remaining[order.price] -= consumed
                        if buy_queue_remaining[order.price] <= 0:
                            del buy_queue_remaining[order.price]
                elif remaining < 0 and mt.price == order.price:
                    if order.price in sell_queue_remaining:
                        consumed = min(mt.quantity, sell_queue_remaining[order.price])
                        mt.quantity -= consumed
                        sell_queue_remaining[order.price] -= consumed
                        if sell_queue_remaining[order.price] <= 0:
                            del sell_queue_remaining[order.price]

                if mt.quantity <= 0:
                    continue

                fill = min(abs(remaining), mt.quantity)
                execution_price = _slippage_adjusted_price(
                    order.price, remaining > 0, config.price_slippage_bps
                )

                if remaining > 0:
                    own_trades.append(
                        InternalTrade(
                            symbol=symbol,
                            price=execution_price,
                            quantity=fill,
                            buyer="SUBMISSION",
                            seller=mt.seller,
                            timestamp=timestamp,
                        )
                    )
                    position[symbol] = position.get(symbol, 0) + fill
                    cash_by_product[symbol] = cash_by_product.get(
                        symbol, 0.0
                    ) - (execution_price * fill)
                    remaining -= fill
                else:
                    own_trades.append(
                        InternalTrade(
                            symbol=symbol,
                            price=execution_price,
                            quantity=fill,
                            buyer=mt.buyer,
                            seller="SUBMISSION",
                            timestamp=timestamp,
                        )
                    )
                    position[symbol] = position.get(symbol, 0) - fill
                    cash_by_product[symbol] = cash_by_product.get(
                        symbol, 0.0
                    ) + (execution_price * fill)
                    remaining += fill

                mt.quantity -= fill

    remaining_market = [t for t in market_available if t.quantity > 0]
    return own_trades, remaining_market


# ─── Trader Invocation ────────────────────────────────────────────────────────


def _build_trading_state(
    trader_data: str,
    tick: TickSnapshot,
    own_trades_prev: Dict[str, List[InternalTrade]],
    market_trades_prev: Dict[str, List[InternalTrade]],
    position: Dict[str, int],
) -> TradingState:
    """Build TradingState exactly like pytrader.rs _build_state()."""
    listings = {
        product: Listing(product, product, "SEASHELLS")
        for product in tick.products
    }

    order_depths: Dict[str, OrderDepth] = {}
    for product, snapshot in tick.products.items():
        depth = OrderDepth()
        for level in snapshot.bids:
            depth.buy_orders[int(level.price)] = int(level.volume)
        for level in snapshot.asks:
            depth.sell_orders[int(level.price)] = -int(level.volume)
        order_depths[product] = depth

    own_trades: Dict[str, List[DmTrade]] = {}
    for sym, trades in own_trades_prev.items():
        own_trades[sym] = [
            DmTrade(t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp)
            for t in trades
        ]

    market_trades: Dict[str, List[DmTrade]] = {}
    for sym, trades in market_trades_prev.items():
        market_trades[sym] = [
            DmTrade(t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp)
            for t in trades
        ]

    pos_dict = {k: int(v) for k, v in position.items()}

    plain = {k: int(v) for k, v in tick.observations.plain.items()}

    conversion_obs: Dict[str, ConversionObservation] = {}
    for product, values in tick.observations.conversion.items():
        conversion_obs[product] = ConversionObservation(
            bidPrice=float(values.get("bidPrice", 0.0)),
            askPrice=float(values.get("askPrice", 0.0)),
            transportFees=float(values.get("transportFees", 0.0)),
            exportTariff=float(values.get("exportTariff", 0.0)),
            importTariff=float(values.get("importTariff", 0.0)),
            sugarPrice=float(values.get("sugarPrice", 0.0)),
            sunlightIndex=float(values.get("sunlightIndex", 0.0)),
        )

    observations = Observation(
        plainValueObservations=plain,
        conversionObservations=conversion_obs,
    )

    return TradingState(
        traderData=str(trader_data),
        timestamp=int(tick.timestamp),
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=pos_dict,
        observations=observations,
    )


def _normalize_run_output(output):
    """Normalize trader.run() output exactly like pytrader.rs _normalize_run_output."""
    if isinstance(output, tuple):
        if len(output) == 3:
            orders, conversions, trader_data = output
        elif len(output) == 2:
            orders, trader_data = output
            conversions = 0
        elif len(output) == 1:
            orders = output[0]
            conversions = 0
            trader_data = ""
        else:
            raise RuntimeError("Trader.run returned tuple with unsupported length")
    else:
        orders = output
        conversions = 0
        trader_data = ""

    if not isinstance(orders, dict):
        raise RuntimeError(f"Trader.run returned non-dict orders: {type(orders)}")

    normalized: Dict[str, List[InternalOrder]] = {}
    for symbol, order_list in orders.items():
        if order_list is None:
            continue
        if not isinstance(symbol, str):
            raise RuntimeError("Orders dictionary keys must be strings")
        if not isinstance(order_list, list):
            raise RuntimeError(f"Orders for {symbol} are not a list")

        norm_orders: List[InternalOrder] = []
        for order in order_list:
            if hasattr(order, "symbol") and hasattr(order, "price") and hasattr(order, "quantity"):
                norm_orders.append(
                    InternalOrder(str(order.symbol), int(order.price), int(order.quantity))
                )
            elif isinstance(order, (tuple, list)) and len(order) == 3:
                norm_orders.append(
                    InternalOrder(str(order[0]), int(order[1]), int(order[2]))
                )
            else:
                raise RuntimeError(f"Unrecognized order type in {symbol}: {order}")
        normalized[symbol] = norm_orders

    return normalized, int(conversions), str(trader_data)


def _invoke_trader(trader, state: TradingState):
    """Call trader.run(state) and capture stdout, like pytrader.rs."""
    captured = io.StringIO()
    with redirect_stdout(captured):
        output = trader.run(state)

    orders, conversions, trader_data = _normalize_run_output(output)
    stdout = captured.getvalue()
    return orders, conversions, trader_data, stdout


# ─── Main Backtest Loop ──────────────────────────────────────────────────────


def run_backtest(request: RunRequest) -> BacktestResult:
    """
    Run the backtest — faithful port of runner.rs run_backtest().
    """
    dataset = load_dataset(request.dataset_file)

    # Filter ticks by requested day
    ticks = [
        tick
        for tick in dataset.ticks
        if request.day is None or tick.day == request.day
    ]
    ticks.sort(key=lambda t: t.timestamp)

    if not ticks:
        raise ValueError("No ticks available for selected dataset/day")

    trader = load_trader(request.trader_file)

    # State tracking
    cash_by_product: Dict[str, float] = {p: 0.0 for p in dataset.products}
    position: Dict[str, int] = {}
    own_trades_prev: Dict[str, List[InternalTrade]] = {}
    market_trades_prev: Dict[str, List[InternalTrade]] = {}
    trader_data = ""

    own_trade_count = 0
    final_pnl_total = 0.0
    final_pnl_by_product = OrderedDict((p, 0.0) for p in dataset.products)

    # Series for visualization
    pnl_series: List[Dict[str, Any]] = []
    position_series: List[Dict[str, Any]] = []
    mid_price_series: List[Dict[str, Any]] = []
    fair_value_series: List[Dict[str, Any]] = []
    market_snapshots: List[Dict[str, Any]] = []
    all_own_trades: List[InternalTrade] = []

    # Historical tracking for microprice
    microprice_history: Dict[str, List[float]] = {p: [] for p in dataset.products}

    tick_count = len(ticks)

    for tick in ticks:
        state = _build_trading_state(
            trader_data, tick, own_trades_prev, market_trades_prev, position
        )

        orders_by_symbol, conversions, trader_data, stdout = _invoke_trader(
            trader, state
        )

        orders_by_symbol, limit_messages = _enforce_position_limits(
            position, orders_by_symbol
        )

        own_trades_tick: Dict[str, List[InternalTrade]] = {}
        market_trades_next: Dict[str, List[InternalTrade]] = {}

        for product in dataset.products:
            snapshot = tick.products.get(product)

            # Build bids and asks sorted correctly
            b = []
            a = []
            if snapshot:
                b = [_BookLevel(l.price, l.volume) for l in snapshot.bids]
                a = [_BookLevel(l.price, l.volume) for l in snapshot.asks]
                b.sort(key=lambda x: -x.price)
                a.sort(key=lambda x: x.price)

            raw_market = tick.market_trades.get(product, [])

            # Build original market trades for next-tick consumption
            original_market: List[InternalTrade] = [
                InternalTrade(
                    symbol=product,
                    price=t.price,
                    quantity=t.quantity,
                    buyer=t.buyer,
                    seller=t.seller,
                    timestamp=t.timestamp if t.timestamp != 0 else tick.timestamp,
                )
                for t in raw_market
            ]

            symbol_own_trades, remaining = _match_orders_for_symbol(
                product,
                orders_by_symbol.get(product, []),
                b,
                a,
                raw_market,
                position,
                cash_by_product,
                tick.timestamp,
                request.matching,
            )

            if symbol_own_trades:
                own_trade_count += len(symbol_own_trades)
                own_trades_tick[product] = symbol_own_trades
                all_own_trades.extend(symbol_own_trades)

            if original_market:
                market_trades_next[product] = original_market

        # Compute Fair Value and PnL
        current_fair_values = {}
        snapshot_entry = {"timestamp": tick.timestamp}
        pnl_by_product = OrderedDict()
        
        for product in dataset.products:
            snapshot = tick.products.get(product)
            snapshot_entry[product] = snapshot

            mid_price = snapshot.mid_price if snapshot else None
            
            # calculate microprice
            microprice = None
            if snapshot and snapshot.bids and snapshot.asks:
                bid_1 = snapshot.bids[0]
                ask_1 = snapshot.asks[0]
                total_vol = bid_1.volume + ask_1.volume
                if total_vol > 0:
                    microprice = (ask_1.volume * bid_1.price + bid_1.volume * ask_1.price) / total_vol
            
            if microprice is None:
                microprice = mid_price

            # fallback fair value base
            fair_val = mid_price
            
            hist = microprice_history[product]
            if microprice is not None:
                if len(hist) > 1:
                    # forward rolling regression on hist (which excludes CURRENT microprice!)
                    n = len(hist)
                    sum_x = sum(range(n))
                    sum_y = sum(hist)
                    sum_x2 = sum(x*x for x in range(n))
                    sum_xy = sum(x * y for x, y in enumerate(hist))
                    
                    denominator = (n * sum_x2 - sum_x * sum_x)
                    if denominator != 0:
                        m = (n * sum_xy - sum_x * sum_y) / denominator
                        c = (sum_y - m * sum_x) / n
                        predicted_microprice = m * n + c
                        fair_val = predicted_microprice
                    else:
                        fair_val = hist[-1]
                elif len(hist) == 1:
                    fair_val = hist[0]
                else:
                    fair_val = microprice
                
                hist.append(microprice)
                if len(hist) > 10:
                    hist.pop(0)

            current_fair_values[product] = fair_val

            mark_to_market = 0.0
            if fair_val is not None:
                mark_to_market = position.get(product, 0) * fair_val
            pnl = cash_by_product.get(product, 0.0) + mark_to_market
            pnl_by_product[product] = pnl

        final_pnl_total = sum(pnl_by_product.values())
        final_pnl_by_product = pnl_by_product

        # Record series data
        pnl_entry = {"timestamp": tick.timestamp, "total": final_pnl_total}
        pnl_entry.update(pnl_by_product)
        pnl_series.append(pnl_entry)

        pos_entry = {"timestamp": tick.timestamp}
        pos_entry.update({p: position.get(p, 0) for p in dataset.products})
        position_series.append(pos_entry)

        mid_entry = {"timestamp": tick.timestamp}
        for product in dataset.products:
            snapshot = tick.products.get(product)
            mid_entry[product] = snapshot.mid_price if snapshot else None
        mid_price_series.append(mid_entry)

        fv_entry = {"timestamp": tick.timestamp}
        fv_entry.update(current_fair_values)
        fair_value_series.append(fv_entry)
        
        market_snapshots.append(snapshot_entry)

        own_trades_prev = own_trades_tick
        market_trades_prev = market_trades_next

    from datetime import datetime, timezone

    metrics = RunMetrics(
        run_id=request.run_id or f"backtest-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        dataset_id=dataset.dataset_id,
        dataset_path=request.dataset_file,
        trader_path=request.trader_file,
        day=request.day,
        matching=request.matching,
        tick_count=tick_count,
        own_trade_count=own_trade_count,
        final_pnl_total=final_pnl_total,
        final_pnl_by_product=final_pnl_by_product,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    return BacktestResult(
        metrics=metrics,
        pnl_series=pnl_series,
        position_series=position_series,
        mid_price_series=mid_price_series,
        fair_value_series=fair_value_series,
        market_snapshots=market_snapshots,
        own_trades_all=all_own_trades,
        products=dataset.products,
    )
