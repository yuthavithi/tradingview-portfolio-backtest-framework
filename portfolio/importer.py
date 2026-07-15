"""
TradingView Excel Importer Module.

This module provides classes and functions to read, parse, and import TradingView
exported backtest Excel (XLSX) reports. It handles different language exports
(English/Chinese), resolves sheet and column names, aggregates trade entry/exit
rows, and constructs standard Trade dataclass objects.
"""

import logging
import os
import glob
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Optional
import pandas as pd

# Setup logger for the module
logger = logging.getLogger("portfolio.importer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class Trade:
    """
    Standard representation of a completed trade parsed from TradingView reports.

    Attributes:
        strategy_name: Name of the strategy.
        trade_id: Unique index/identifier for the trade.
        entry_time: Timestamp when the position was opened.
        exit_time: Timestamp when the position was closed.
        side: Position side ('Long' or 'Short').
        entry_price: Weighted average entry fill price.
        exit_price: Weighted average exit fill price.
        contracts: Total number of contracts/shares traded.
        position_value: Position value at entry (entry_price * contracts).
        commission: Total commission paid for the trade (entry + exit).
        profit: Realized profit/loss.
        profit_percent: Realized profit/loss percentage.
        holding_time: Duration of the trade (timedelta).
    """

    strategy_name: str
    trade_id: Union[int, str]
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    contracts: float
    position_value: float
    commission: float
    profit: float
    profit_percent: float
    holding_time: timedelta
    mae: float = 0.0
    initial_capital: Optional[float] = None



@dataclass
class TradingViewReport:
    """
    A representation of a single TradingView exported backtest report.

    Attributes:
        file_path: The filesystem path of the XLSX file.
        strategy_name: Extracted or overridden strategy name.
        properties: Raw or parsed properties key-value pairs.
        performance_summary: Raw performance summary data.
        trades: List of parsed Trade objects.
        orders: Raw orders log data if present.
    """

    file_path: str
    strategy_name: str
    properties: Dict[str, Any]
    performance_summary: pd.DataFrame
    trades: List[Trade]
    orders: pd.DataFrame


def normalize_column_name(col: str) -> str:
    """
    Normalizes a column header to lowercase, strips trailing/leading space,
    and removes common symbols/parentheses to facilitate matching.
    """
    col_clean = str(col).lower().strip()
    for char in ["(", ")", "[", "]", "$", "%", "（", "）", "「", "」"]:
        col_clean = col_clean.replace(char, "")
    return " ".join(col_clean.split())


def find_column_by_candidates(cols: List[str], candidates: List[str]) -> Optional[str]:
    """
    Searches a list of column names for the best match against a list of candidate strings.
    First does exact matching on normalized values, then checks prefix/substring matches.
    """
    # Exact normalized match
    for col in cols:
        norm = normalize_column_name(col)
        if norm in candidates:
            return col

    # Substring match (candidate in column name)
    for col in cols:
        norm = normalize_column_name(col)
        for cand in candidates:
            if cand in norm or norm.startswith(cand):
                return col
    return None


def resolve_sheet_name(sheet_names: List[str], candidates: List[str]) -> Optional[str]:
    """
    Searches a list of sheet names for a case-insensitive match against candidates.
    """
    for name in sheet_names:
        if name.lower().strip() in candidates:
            return name
    return None


def parse_tradingview_file(
    file_path: str,
    strategy_name_override: Optional[str] = None,
    default_commission_rate: Optional[float] = None,
) -> TradingViewReport:
    """
    Parses a single TradingView exported Excel (.xlsx) file.

    Args:
        file_path: Path to the XLSX file.
        strategy_name_override: Optional string to override the strategy name.
        default_commission_rate: Optional rate to use instead of reading properties.

    Returns:
        A TradingViewReport instance containing parsed sheets and Trades.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Parsing TradingView report: {file_path}")
    xls = pd.ExcelFile(file_path)

    # 1. Properties sheet resolution
    prop_sheet = resolve_sheet_name(
        xls.sheet_names, ["properties", "属性", "settings", "设置"]
    )
    properties = {}
    if prop_sheet:
        df_prop = xls.parse(prop_sheet)
        if not df_prop.empty and len(df_prop.columns) >= 2:
            name_col = df_prop.columns[0]
            val_col = df_prop.columns[1]
            for _, row in df_prop.iterrows():
                k = str(row[name_col]).strip()
                v = row[val_col]
                properties[k.lower()] = v
    else:
        logger.warning(f"No properties sheet found in {file_path}")

    # Extract symbol
    symbol = (
        properties.get("symbol")
        or properties.get("商品代码")
        or properties.get("ticker")
        or ""
    )

    # Determine strategy name
    if strategy_name_override:
        strategy_name = strategy_name_override
    else:
        # Default strategy name logic from filename
        filename = os.path.basename(file_path)
        strategy_name = os.path.splitext(filename)[0]
        if symbol:
            symbol_norm = str(symbol).replace(":", "_")
            if symbol_norm in filename:
                parts = filename.split(symbol_norm)
                strategy_name = parts[0].strip("_")

    # Extract commission settings
    commission_rate = 0.0
    if default_commission_rate is not None:
        commission_rate = default_commission_rate
    else:
        comm_val = properties.get("commission") or properties.get("佣金")
        if comm_val is not None:
            try:
                # E.g., if properties sheet has '0.05' as a float
                commission_rate = float(comm_val) / 100.0
            except ValueError:
                logger.warning(
                    f"Could not parse commission rate '{comm_val}' as float. Defaulting to 0."
                )

    # Extract initial capital
    strategy_initial_capital = None
    init_cap_val = properties.get("initial capital") or properties.get("初始资金")
    if init_cap_val is not None:
        try:
            strategy_initial_capital = float(init_cap_val)
        except ValueError:
            pass

    # 2. Performance Summary sheet resolution
    perf_sheet = resolve_sheet_name(
        xls.sheet_names,
        [
            "performance",
            "表现",
            "performance summary",
            "交易分析",
            "overview",
            "总览",
        ],
    )
    df_perf = pd.DataFrame()
    if perf_sheet:
        df_perf = xls.parse(perf_sheet)
    else:
        logger.warning(f"No performance summary sheet found in {file_path}")

    # 3. Orders sheet resolution
    orders_sheet = resolve_sheet_name(
        xls.sheet_names, ["orders", "订单", "list of orders", "订单列表"]
    )
    df_orders = pd.DataFrame()
    if orders_sheet:
        df_orders = xls.parse(orders_sheet)
    else:
        logger.debug(f"No orders log sheet found in {file_path}")

    # 4. Trades sheet resolution
    trades_sheet = resolve_sheet_name(
        xls.sheet_names, ["trades", "交易", "list of trades", "交易列表"]
    )
    trades_list: List[Trade] = []

    if trades_sheet:
        df_trades = xls.parse(trades_sheet)
        cols = list(df_trades.columns)

        # Map columns using candidates
        c_id = find_column_by_candidates(
            cols, ["trade number", "trade #", "trade id", "id", "编号", "交易编号"]
        )
        c_type = find_column_by_candidates(cols, ["type", "action", "类型"])
        c_time = find_column_by_candidates(
            cols,
            [
                "date and time",
                "date/time",
                "time",
                "datetime",
                "date",
                "时间",
                "日期和时间",
            ],
        )
        c_price = find_column_by_candidates(
            cols,
            [
                "price usdt",
                "price usd",
                "price",
                "execution price",
                "fill price",
                "价格 usdt",
                "价格 usd",
                "价格",
            ],
        )
        c_qty = find_column_by_candidates(
            cols,
            ["size qty", "contracts", "quantity", "qty", "size", "数量", "大小（数量）"],
        )
        c_profit = find_column_by_candidates(
            cols,
            [
                "net pnl usdt",
                "net pnl usd",
                "net pnl",
                "profit",
                "net profit",
                "realized pnl",
                "pnl",
                "损益",
                "净损益 usdt",
                "净损益 usd",
                "净损益",
            ],
        )
        c_pct = find_column_by_candidates(
            cols, ["return", "profit percent", "return percent", "profit_percent", "回报", "回报 %"]
        )
        c_comm = find_column_by_candidates(cols, ["commission", "佣金"])
        c_mae = find_column_by_candidates(
            cols,
            [
                "mae",
                "adverse excursion",
                "最大不利偏移",
            ],
        )

        required_cols = [c_id, c_type, c_time, c_price, c_qty]
        if not all(required_cols):
            missing = [
                name
                for name, col in zip(
                    ["id", "type", "time", "price", "qty"], required_cols
                )
                if col is None
            ]
            raise ValueError(
                f"Missing required columns in sheet {trades_sheet}: {missing}"
            )

        # Drop rows with missing trade ID
        df_trades = df_trades.dropna(subset=[c_id])

        # Group by trade ID and process
        grouped = df_trades.groupby(c_id)
        for tid, group in grouped:
            entries = []
            exits = []

            for _, row in group.iterrows():
                t_val = str(row[c_type]).lower()

                # Robust entry vs exit checks
                is_entry = (
                    "entry" in t_val
                    or "进场" in t_val
                    or "buy" in t_val
                    or "sell" in t_val
                ) and ("exit" not in t_val and "出场" not in t_val)

                is_exit = (
                    "exit" in t_val
                    or "出场" in t_val
                    or "close" in t_val
                    or "sl" in t_val
                    or "tp" in t_val
                )

                if not is_entry and not is_exit:
                    # Fallback to simple direction
                    if "long" in t_val or "short" in t_val:
                        is_entry = True

                if is_entry:
                    entries.append(row)
                elif is_exit:
                    exits.append(row)

            # Skip incomplete trades
            if not entries or not exits:
                logger.debug(f"Trade ID {tid} is incomplete. Entries: {len(entries)}, Exits: {len(exits)}")
                continue

            # Sort entries and exits chronologically
            entries_sorted = sorted(entries, key=lambda r: pd.to_datetime(r[c_time]))
            exits_sorted = sorted(exits, key=lambda r: pd.to_datetime(r[c_time]))

            entry_time = pd.to_datetime(entries_sorted[0][c_time])
            exit_time = pd.to_datetime(exits_sorted[-1][c_time])

            # Sum entry and exit sizes
            total_entry_qty = sum(float(r[c_qty]) for r in entries_sorted)
            total_exit_qty = sum(float(r[c_qty]) for r in exits_sorted)

            # Compute weighted average entry and exit prices
            entry_price = (
                sum(float(r[c_price]) * float(r[c_qty]) for r in entries_sorted)
                / total_entry_qty
                if total_entry_qty > 0
                else 0.0
            )
            exit_price = (
                sum(float(r[c_price]) * float(r[c_qty]) for r in exits_sorted)
                / total_exit_qty
                if total_exit_qty > 0
                else 0.0
            )

            contracts = total_entry_qty

            # Determine side from first entry order type
            first_entry_type = str(entries_sorted[0][c_type]).lower()
            side = (
                "Long"
                if "long" in first_entry_type
                or "buy" in first_entry_type
                or "多头" in first_entry_type
                else "Short"
            )

            # Compute position value at entry
            position_value = entry_price * contracts

            # Get profit and profit percentage
            profit = sum(float(r[c_profit]) for r in exits_sorted) if c_profit else 0.0
            profit_percent = float(exits_sorted[-1][c_pct]) if c_pct else 0.0

            # MAE is typically negative or zero. Take the min value from exits.
            mae = 0.0
            if c_mae:
                mae_vals = [float(r[c_mae]) for r in exits_sorted if not pd.isna(r[c_mae])]
                if mae_vals:
                    mae = min(mae_vals)

            # Commission calculation
            # Try to sum from commission column first
            if c_comm and not pd.isna(exits_sorted[-1][c_comm]):
                commission = sum(
                    float(r[c_comm]) for r in (entries_sorted + exits_sorted) if not pd.isna(r[c_comm])
                )
            else:
                # Apply rate to entry value and exit value
                commission = (
                    entry_price * total_entry_qty + exit_price * total_exit_qty
                ) * commission_rate

            holding_time = exit_time - entry_time

            # Handle parsing trade ID type correctly
            try:
                numeric_tid: Union[int, str] = int(tid)
            except ValueError:
                numeric_tid = str(tid)

            trade = Trade(
                strategy_name=strategy_name,
                trade_id=numeric_tid,
                entry_time=entry_time,
                exit_time=exit_time,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                contracts=contracts,
                position_value=position_value,
                commission=commission,
                profit=profit,
                profit_percent=profit_percent,
                holding_time=holding_time,
                mae=mae,
                initial_capital=strategy_initial_capital,
            )
            trades_list.append(trade)

        # Sort trades list by trade id numerically
        trades_list.sort(key=lambda t: t.trade_id if isinstance(t.trade_id, int) else 0)
    else:
        logger.warning(f"No trades sheet found in {file_path}")

    xls.close()
    return TradingViewReport(
        file_path=file_path,
        strategy_name=strategy_name,
        properties=properties,
        performance_summary=df_perf,
        trades=trades_list,
        orders=df_orders,
    )


def import_tradingview_files(
    file_paths: Union[str, List[str]],
    strategy_name_override: Optional[str] = None,
    default_commission_rate: Optional[float] = None,
) -> List[Trade]:
    """
    Imports and parses multiple TradingView exported Excel files, combining
    all parsed Trade objects into a single flat list.

    Args:
        file_paths: A single file path, list of file paths, or glob pattern/directory path.
        strategy_name_override: Optional override for the strategy name.
        default_commission_rate: Optional rate to use instead of reading properties.

    Returns:
        A combined list of Trade objects from all parsed reports.
    """
    resolved_paths: List[str] = []

    if isinstance(file_paths, str):
        # Check if it's a directory
        if os.path.isdir(file_paths):
            xlsx_pattern = os.path.join(file_paths, "*.xlsx")
            resolved_paths = glob.glob(xlsx_pattern)
        # Check if it's a glob pattern
        elif "*" in file_paths or "?" in file_paths:
            resolved_paths = glob.glob(file_paths)
        else:
            resolved_paths = [file_paths]
    elif isinstance(file_paths, list):
        for path in file_paths:
            # Recursively handle items in the list if they are directories or globs
            if os.path.isdir(path):
                resolved_paths.extend(glob.glob(os.path.join(path, "*.xlsx")))
            elif "*" in path or "?" in path:
                resolved_paths.extend(glob.glob(path))
            else:
                resolved_paths.append(path)

    # Filter out lock files or temporary files (e.g. starting with .~ or ~$)
    resolved_paths = [
        p
        for p in resolved_paths
        if not os.path.basename(p).startswith("~$")
        and not os.path.basename(p).startswith(".~")
        and p.endswith(".xlsx")
    ]

    all_trades: List[Trade] = []
    seen_strategies: Dict[str, int] = {}
    for path in resolved_paths:
        try:
            report = parse_tradingview_file(
                file_path=path,
                strategy_name_override=strategy_name_override,
                default_commission_rate=default_commission_rate,
            )
            
            strat_name = report.strategy_name
            
            # Extract symbol and clean exchange prefix (e.g. BINANCE:BTCUSDT.P -> BTCUSDT.P)
            symbol = report.properties.get("symbol") or report.properties.get("商品代码") or ""
            if symbol and ":" in str(symbol):
                symbol = str(symbol).split(":")[-1]
                
            if symbol:
                unique_name = f"{strat_name} ({symbol})"
            else:
                unique_name = strat_name
                
            if unique_name in seen_strategies:
                seen_strategies[unique_name] += 1
                unique_name = f"{unique_name}_{seen_strategies[unique_name]}"
            else:
                seen_strategies[unique_name] = 1
                
            for t in report.trades:
                t.strategy_name = unique_name
                
            all_trades.extend(report.trades)
        except Exception as e:
            logger.error(f"Error parsing file {path}: {str(e)}")
            raise e

    return all_trades
