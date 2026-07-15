"""
Shared Capital Simulation Engine Module.

This module implements the event-driven SharedCapitalEngine, where multiple
TradingView strategies share a single account with leverage-based margin checks.
It tracks Cash, Equity, Margin Used, Floating PnL, and Available Margin over time,
and calculates performance statistics and drawdown curves.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import copy

from portfolio.importer import Trade
from portfolio.events import BaseEvent
from portfolio.queue import EventQueue
from portfolio.risk import RiskEngine

# Setup logger for the module
logger = logging.getLogger("portfolio.shared_engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class MarginRequestEvent(BaseEvent):
    """
    Event representing a trade's entry and request for margin allocation.
    """
    trade_id: Union[int, str]
    position_value: float
    profit: float
    side: str
    strategy_name: str
    trade: Trade
    scale_factor: float = 1.0


@dataclass
class MarginReleaseEvent(BaseEvent):
    """
    Event representing a trade's exit and release of margin.
    """
    trade_id: Union[int, str]
    position_value: float
    profit: float
    side: str
    strategy_name: str
    trade: Trade


class SharedCapitalEngine:
    """
    Simulates a shared capital account backtest where multiple strategies run concurrently.
    Allocates margin dynamically and tracks account metrics.
    """

    def __init__(
        self,
        initial_equity: float = 1000.0,
        leverage: float = 10.0,
        strategy_weights: Optional[Dict[str, float]] = None,
        stress_test_drawdown: bool = False,
    ) -> None:
        """
        Initializes the Shared Capital Engine.

        Args:
            initial_equity: The starting account capital in USDT.
            leverage: The leverage factor (e.g. 10.0 for 10x).
        """
        self.initial_equity = initial_equity
        self.leverage = leverage
        self.stress_test_drawdown = stress_test_drawdown
        self.strategy_weights = strategy_weights or {}
        self.queue = EventQueue()

        # Engine state variables
        self.cash = initial_equity
        self.equity = initial_equity
        self.margin_used = 0.0
        self.floating_pnl = 0.0
        self.available_margin = initial_equity

        # Track active and skipped trades
        self.active_trades: Dict[Union[int, str], Dict[str, Any]] = {}
        self.executed_trade_ids: List[Union[int, str]] = []
        self.skipped_trade_ids: List[Union[int, str]] = []
        self.conflict_logs: List[Dict[str, Any]] = []

        # History of state snapshots
        self.history: List[Dict[str, Any]] = []
        self.risk_engine: Optional[RiskEngine] = None
        self.scaled_trades: List[Trade] = []

        # Strategy-level unscaled equity tracking
        self.strategy_realized_profits: Dict[str, float] = {}
        self.strategy_active_trades: Dict[str, List[Trade]] = {}

    def _reset_state(self) -> None:
        """Resets simulation state variables."""
        self.cash = self.initial_equity
        self.equity = self.initial_equity
        self.margin_used = 0.0
        self.floating_pnl = 0.0
        self.available_margin = self.initial_equity
        self.active_trades.clear()
        self.executed_trade_ids.clear()
        self.skipped_trade_ids.clear()
        self.conflict_logs.clear()
        self.history.clear()
        self.queue.clear()
        self.scaled_trades.clear()
        self.strategy_realized_profits.clear()
        self.strategy_active_trades.clear()
        if self.risk_engine is not None:
            self.risk_engine.reset()

    def run(self, trades: List[Trade], risk_engine: Optional[RiskEngine] = None) -> Dict[str, Any]:
        """
        Executes the event-driven shared capital simulation on the given trades list.

        Args:
            trades: List of Trade dataclass objects.
            risk_engine: Optional RiskEngine instance to enforce limits.

        Returns:
            A dictionary containing backtest results and performance metrics.
        """
        self.risk_engine = risk_engine
        self._reset_state()

        if not trades:
            logger.warning("No trades provided for simulation.")
            return self._generate_empty_results()

        # 1. Build and queue events
        for i, trade in enumerate(trades):
            # Create a unique trade ID across different strategies and files
            strat = trade.strategy_name or "Strat"
            tid = f"{strat}_{trade.trade_id}_{i}" if trade.trade_id is not None else f"{strat}_T_{i}"
            
            entry_evt = MarginRequestEvent(
                timestamp=trade.entry_time,
                trade_id=tid,
                position_value=trade.position_value,
                profit=trade.profit,
                side=trade.side,
                strategy_name=strat,
                trade=trade,
            )
            
            # Avoid same-timestamp entry/exit ordering issues by adding a microsecond to exits of 0-duration trades
            exit_timestamp = trade.exit_time
            if exit_timestamp == trade.entry_time:
                exit_timestamp = trade.entry_time + timedelta(microseconds=1)
                
            exit_evt = MarginReleaseEvent(
                timestamp=exit_timestamp,
                trade_id=tid,
                position_value=trade.position_value,
                profit=trade.profit,
                side=trade.side,
                strategy_name=strat,
                trade=trade,
            )
            self.queue.push(entry_evt)
            self.queue.push(exit_evt)

        # 2. Record initial history state
        start_time = min(t.entry_time for t in trades) - timedelta(seconds=1)
        self._record_snapshot(start_time)

        # 3. Main event loop
        while not self.queue.empty():
            peek_evt = self.queue.peek()
            if peek_evt is None:
                break
            current_time = peek_evt.timestamp

            # Update floating PnL and equity at this timestamp
            self._update_floating_pnl(current_time)

            # Gather all events at this timestamp
            events_at_time = []
            while not self.queue.empty():
                next_evt = self.queue.peek()
                if next_evt is not None and next_evt.timestamp == current_time:
                    events_at_time.append(self.queue.pop())
                else:
                    break

            # Separate events by type
            releases = [e for e in events_at_time if isinstance(e, MarginReleaseEvent)]
            requests = [e for e in events_at_time if isinstance(e, MarginRequestEvent)]
            others = [e for e in events_at_time if not isinstance(e, (MarginReleaseEvent, MarginRequestEvent))]

            # 1. Process all releases first to free up margin
            for event in releases:
                self._process_event(event)

            # Recalculate floating PnL and equity after exits are processed
            self._update_floating_pnl(current_time)

            # 1.5 Scale requests based on current equity
            for event in requests:
                self._update_floating_pnl(current_time)
                weight = self.strategy_weights.get(event.strategy_name, 1.0)
                strategy_initial = event.trade.initial_capital
                
                # Add to strategy active trades tracking (for unscaled strat equity calculation)
                self.strategy_active_trades.setdefault(event.strategy_name, []).append(event.trade)
                
                if strategy_initial and strategy_initial > 0:
                    # Calculate strategy's own unscaled equity at this time
                    strat_floating = self._get_strategy_floating_profit(event.strategy_name, current_time)
                    strat_realized = self.strategy_realized_profits.get(event.strategy_name, 0.0)
                    strat_equity = max(1.0, strategy_initial + strat_realized + strat_floating)
                    
                    # scale_factor = (portfolio_equity * weight) / strategy_equity
                    scale_factor = (self.equity * weight) / strat_equity
                else:
                    scale_factor = weight
                    
                event.scale_factor = scale_factor
                event.position_value *= scale_factor
                event.profit *= scale_factor

            # 2. Process requests. Check for multi-strategy conflict
            unique_strategies = set(e.strategy_name for e in requests)
            if len(requests) >= 2 and len(unique_strategies) >= 2:
                # Capture available margin BEFORE resolving conflict
                avail_margin_before = self.available_margin
                for event in requests:
                    required_margin = event.position_value / self.leverage
                    
                    allowed_by_risk = True
                    if self.risk_engine is not None:
                        allowed_by_risk = self.risk_engine.validate_trade_entry(
                            ticker=event.strategy_name,
                            quantity=event.trade.contracts,
                            price=event.trade.entry_price,
                            equity=self.equity,
                            active_positions_count=len(set(t["trade"].strategy_name for t in self.active_trades.values())),
                            current_margin_used=self.margin_used,
                            current_portfolio_risk=0.0,
                            timestamp=event.timestamp
                        )
                    
                    if allowed_by_risk and self.available_margin >= required_margin:
                        # Execute Trade
                        self.active_trades[event.trade_id] = {
                            "required_margin": required_margin,
                            "profit": event.profit,
                            "trade": event.trade,
                            "scale_factor": event.scale_factor,
                        }
                        
                        scaled_trade = copy.deepcopy(event.trade)
                        scaled_trade.contracts *= event.scale_factor
                        scaled_trade.position_value *= event.scale_factor
                        scaled_trade.profit *= event.scale_factor
                        scaled_trade.commission *= event.scale_factor
                        self.scaled_trades.append(scaled_trade)
                        
                        self.margin_used += required_margin
                        self.available_margin = self.equity - self.margin_used
                        self.executed_trade_ids.append(event.trade_id)
                        logger.debug(
                            f"[{event.timestamp}] CONFLICT-WIN: Trade {event.trade_id} from {event.strategy_name}. "
                            f"Required Margin: ${required_margin:.2f}, Avail Margin: ${self.available_margin:.2f}"
                        )
                        self.conflict_logs.append({
                            "conflict_time": current_time,
                            "strategy": event.strategy_name,
                            "required_margin": required_margin,
                            "available_margin": avail_margin_before,
                            "winner": True,
                            "loser": False,
                            "skipped_trade": None
                        })
                    else:
                        # Skip Trade
                        self.skipped_trade_ids.append(event.trade_id)
                        logger.debug(
                            f"[{event.timestamp}] CONFLICT-LOSS: Trade {event.trade_id} from {event.strategy_name}. "
                            f"Required Margin: ${required_margin:.2f} > Avail Margin: ${self.available_margin:.2f}"
                        )
                        self.conflict_logs.append({
                            "conflict_time": current_time,
                            "strategy": event.strategy_name,
                            "required_margin": required_margin,
                            "available_margin": avail_margin_before,
                            "winner": False,
                            "loser": True,
                            "skipped_trade": event.trade
                        })
            else:
                # Process requests normally
                for event in requests:
                    self._process_event(event)

            # 3. Process any other event types
            for event in others:
                self._process_event(event)

            # Record snapshot for this timestamp
            self._record_snapshot(current_time)

        # 4. Generate curves and statistics
        results = self._generate_metrics(trades)
        results["scaled_trades"] = self.scaled_trades
        return results

    def _process_event(self, event: BaseEvent) -> None:
        """Processes a single simulation event."""
        if isinstance(event, MarginRequestEvent):
            required_margin = event.position_value / self.leverage
            
            allowed_by_risk = True
            if self.risk_engine is not None:
                allowed_by_risk = self.risk_engine.validate_trade_entry(
                    ticker=event.strategy_name,
                    quantity=event.trade.contracts,
                    price=event.trade.entry_price,
                    equity=self.equity,
                    active_positions_count=len(set(t["trade"].strategy_name for t in self.active_trades.values())),
                    current_margin_used=self.margin_used,
                    current_portfolio_risk=0.0,
                    timestamp=event.timestamp
                )
            
            if allowed_by_risk and self.available_margin >= required_margin:
                # Execute Trade
                self.active_trades[event.trade_id] = {
                    "required_margin": required_margin,
                    "profit": event.profit,
                    "trade": event.trade,
                    "scale_factor": event.scale_factor,
                }
                
                scaled_trade = copy.deepcopy(event.trade)
                scaled_trade.contracts *= event.scale_factor
                scaled_trade.position_value *= event.scale_factor
                scaled_trade.profit *= event.scale_factor
                scaled_trade.commission *= event.scale_factor
                self.scaled_trades.append(scaled_trade)
                
                self.margin_used += required_margin
                self.available_margin = self.equity - self.margin_used
                self.executed_trade_ids.append(event.trade_id)
                logger.debug(
                    f"[{event.timestamp}] EXECUTED: Trade {event.trade_id} from {event.strategy_name}. "
                    f"Required Margin: ${required_margin:.2f}, Avail Margin: ${self.available_margin:.2f}"
                )
            else:
                # Skip Trade
                self.skipped_trade_ids.append(event.trade_id)
                if not allowed_by_risk:
                    logger.debug(
                        f"[{event.timestamp}] RISK-BLOCKED: Trade {event.trade_id} from {event.strategy_name} "
                        f"blocked by risk limits."
                    )
                else:
                    logger.debug(
                        f"[{event.timestamp}] SKIPPED: Trade {event.trade_id} from {event.strategy_name}. "
                        f"Required Margin: ${required_margin:.2f} > Avail Margin: ${self.available_margin:.2f}"
                    )

        elif isinstance(event, MarginReleaseEvent):
            if event.trade_id in self.active_trades:
                # Release Margin and Realize PnL
                info = self.active_trades.pop(event.trade_id)
                required_margin = info["required_margin"]
                scale_factor = info.get("scale_factor", 1.0)
                
                scaled_profit = event.profit * scale_factor
                
                self.margin_used -= required_margin
                self.cash += scaled_profit
                
                # Update strategy realized profits and remove from active trades
                self.strategy_realized_profits[event.strategy_name] = (
                    self.strategy_realized_profits.get(event.strategy_name, 0.0) + event.profit
                )
                if event.strategy_name in self.strategy_active_trades:
                    # Remove the trade from active list
                    self.strategy_active_trades[event.strategy_name] = [
                        t for t in self.strategy_active_trades[event.strategy_name]
                        if t.trade_id != event.trade.trade_id
                    ]
                
                self._update_floating_pnl(event.timestamp)
                
                logger.debug(
                    f"[{event.timestamp}] EXIT: Trade {event.trade_id} from {event.strategy_name}. "
                    f"Scaled Profit: ${scaled_profit:.2f}, Equity: ${self.equity:.2f}"
                )
                if self.risk_engine is not None:
                    self.risk_engine.update_daily_metrics(event.timestamp, self.equity, scaled_profit)

    def _record_snapshot(self, timestamp: datetime) -> None:
        """Records a snapshot of the account state."""
        self.history.append({
            "timestamp": timestamp,
            "cash": self.cash,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "floating_pnl": self.floating_pnl,
            "available_margin": self.available_margin,
            "concurrent_positions": len(set(t["trade"].strategy_name for t in self.active_trades.values())),
        })

    def _update_floating_pnl(self, current_time: datetime) -> None:
        """Calculates floating PnL of active trades and updates equity/margin."""
        total_floating_pnl = 0.0
        for trade_id, info in self.active_trades.items():
            trade = info["trade"]
            scale_factor = info.get("scale_factor", 1.0)
            
            if self.stress_test_drawdown:
                # Apply the maximum adverse excursion (MAE) instantly
                total_floating_pnl += (trade.mae * scale_factor)
            else:
                duration = (trade.exit_time - trade.entry_time).total_seconds()
                if duration > 0:
                    elapsed = (current_time - trade.entry_time).total_seconds()
                    elapsed = max(0.0, min(elapsed, duration))
                    total_floating_pnl += (trade.profit * scale_factor) * (elapsed / duration)
        
        self.floating_pnl = total_floating_pnl
        self.equity = self.cash + self.floating_pnl
        self.available_margin = self.equity - self.margin_used

    def _get_strategy_floating_profit(self, strategy_name: str, current_time: datetime) -> float:
        """Calculates unscaled floating profit for a specific strategy."""
        floating_profit = 0.0
        active_list = self.strategy_active_trades.get(strategy_name, [])
        for trade in active_list:
            if self.stress_test_drawdown:
                floating_profit += trade.mae
            else:
                duration = (trade.exit_time - trade.entry_time).total_seconds()
                if duration > 0:
                    elapsed = (current_time - trade.entry_time).total_seconds()
                    elapsed = max(0.0, min(elapsed, duration))
                    floating_profit += trade.profit * (elapsed / duration)
        return floating_profit

    def _generate_metrics(self, all_trades: List[Trade]) -> Dict[str, Any]:
        """Generates performance curves, stats, drawdown, monthly returns, CAGR."""
        # 1. Equity and Drawdown Curves
        equity_curve = []
        drawdown_curve = []
        peak = self.initial_equity

        for snap in self.history:
            eq = snap["equity"]
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0.0
            
            equity_curve.append((snap["timestamp"], eq))
            drawdown_curve.append((snap["timestamp"], dd * 100.0))

        # 2. Maximum Drawdown
        max_drawdown = max([dd for _, dd in drawdown_curve]) if drawdown_curve else 0.0

        # 3. CAGR
        if len(self.history) < 2:
            cagr = 0.0
        else:
            start_val = self.initial_equity
            end_val = self.history[-1]["equity"]
            start_date = self.history[0]["timestamp"]
            end_date = self.history[-1]["timestamp"]
            duration_days = (end_date - start_date).total_seconds() / 86400.0
            years = duration_days / 365.25
            if years > (1.0 / 365.25) and start_val > 0 and end_val > 0:
                try:
                    cagr = (end_val / start_val) ** (1.0 / years) - 1.0
                except (OverflowError, ZeroDivisionError):
                    cagr = float("inf")
            else:
                cagr = 0.0

        # 4. Monthly Returns
        monthly_returns = self._calculate_monthly_returns()

        # 5. Trade Statistics (calculated on executed scaled trades)
        trade_stats = self._calculate_trade_statistics(all_trades)

        # 6. Conflict metrics
        conflict_timestamps = set(log["conflict_time"] for log in self.conflict_logs)
        conflict_frequency = len(conflict_timestamps)
        
        total_requested = len(all_trades)
        conflict_rate = conflict_frequency / total_requested if total_requested > 0 else 0.0
        
        skipped_profit = 0.0
        avoided_loss = 0.0
        for log in self.conflict_logs:
            if log["loser"] and log["skipped_trade"] is not None:
                profit = log["skipped_trade"].profit
                if profit > 0:
                    skipped_profit += profit
                elif profit < 0:
                    avoided_loss += abs(profit)

        # Margin and position usage metrics
        margin_usages = [h["margin_used"] for h in self.history] if self.history else [0.0]
        positions = [h["concurrent_positions"] for h in self.history] if self.history else [0]
        
        peak_margin_usage = max(margin_usages)
        maximum_concurrent_positions = max(positions)
        
        average_margin_usage = sum(margin_usages) / len(margin_usages) if margin_usages else 0.0
        average_concurrent_positions = sum(positions) / len(positions) if positions else 0.0
        
        total_seconds = 0.0
        weighted_margin = 0.0
        weighted_positions = 0.0
        for i in range(len(self.history) - 1):
            snap_curr = self.history[i]
            snap_next = self.history[i+1]
            duration = (snap_next["timestamp"] - snap_curr["timestamp"]).total_seconds()
            if duration > 0:
                total_seconds += duration
                weighted_margin += snap_curr["margin_used"] * duration
                weighted_positions += snap_curr["concurrent_positions"] * duration
                
        if total_seconds > 0:
            time_weighted_average_margin_usage = weighted_margin / total_seconds
            time_weighted_average_concurrent_positions = weighted_positions / total_seconds
        else:
            time_weighted_average_margin_usage = average_margin_usage
            time_weighted_average_concurrent_positions = average_concurrent_positions
            
        margin_efficiency = average_margin_usage / peak_margin_usage if peak_margin_usage > 0 else 0.0
        time_weighted_margin_efficiency = time_weighted_average_margin_usage / peak_margin_usage if peak_margin_usage > 0 else 0.0
        
        capital_efficiency = average_margin_usage / self.initial_equity if self.initial_equity > 0 else 0.0
        time_weighted_capital_efficiency = time_weighted_average_margin_usage / self.initial_equity if self.initial_equity > 0 else 0.0

        metrics = {
            "initial_equity": self.initial_equity,
            "leverage": self.leverage,
            "ending_equity": self.equity,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "monthly_returns": monthly_returns,
            "trade_statistics": trade_stats,
            "conflict_report": self.conflict_logs,
            "conflict_frequency": conflict_frequency,
            "conflict_rate": conflict_rate,
            "skipped_profit": skipped_profit,
            "avoided_loss": avoided_loss,
            "capital_efficiency": capital_efficiency,
            "margin_efficiency": margin_efficiency,
            "time_weighted_capital_efficiency": time_weighted_capital_efficiency,
            "time_weighted_margin_efficiency": time_weighted_margin_efficiency,
            "average_margin_usage": average_margin_usage,
            "time_weighted_average_margin_usage": time_weighted_average_margin_usage,
            "peak_margin_usage": peak_margin_usage,
            "average_concurrent_positions": average_concurrent_positions,
            "time_weighted_average_concurrent_positions": time_weighted_average_concurrent_positions,
            "maximum_concurrent_positions": maximum_concurrent_positions,
        }

        if self.risk_engine is not None:
            metrics["risk_report"] = self.risk_engine.generate_report(self.initial_equity, self.equity)
            metrics["risk_parameters"] = self.risk_engine.params

        return metrics

    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculates percentage returns grouped by calendar month."""
        if not self.history:
            return {}

        # Get last equity snapshot of each month
        monthly_snapshots: Dict[tuple, float] = {}
        for snap in self.history:
            dt = snap["timestamp"]
            key = (dt.year, dt.month)
            monthly_snapshots[key] = snap["equity"]

        start_date = self.history[0]["timestamp"]
        end_date = self.history[-1]["timestamp"]

        curr_year = start_date.year
        curr_month = start_date.month

        # Generate month range keys chronologically
        keys = []
        while (curr_year, curr_month) <= (end_date.year, end_date.month):
            keys.append((curr_year, curr_month))
            curr_month += 1
            if curr_month > 12:
                curr_month = 1
                curr_year += 1

        # Resolve equity sequence for the months
        last_equity = self.initial_equity
        monthly_equities = {}
        for key in keys:
            if key in monthly_snapshots:
                last_equity = monthly_snapshots[key]
            monthly_equities[key] = last_equity

        # Calculate monthly percentage returns
        monthly_returns = {}
        prev_equity = self.initial_equity
        for key in keys:
            curr_equity = monthly_equities[key]
            ret = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            monthly_returns[f"{key[0]}-{key[1]:02d}"] = ret * 100.0
            prev_equity = curr_equity

        return monthly_returns

    def _calculate_trade_statistics(self, all_trades: List[Trade]) -> Dict[str, Any]:
        """Calculates granular trade metrics for executed vs skipped trades."""
        # Use dynamically scaled trades for accurate profit/loss metrics
        executed_trades = self.scaled_trades

        total_count = len(all_trades)
        executed_count = len(executed_trades)
        skipped_count = len(self.skipped_trade_ids)

        if executed_count == 0:
            return {
                "total_trades": total_count,
                "executed_trades": 0,
                "skipped_trades": skipped_count,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "profit_factor": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        wins = [t.profit for t in executed_trades if t.profit > 0]
        losses = [t.profit for t in executed_trades if t.profit < 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / executed_count) * 100.0

        net_profit = sum(t.profit for t in executed_trades)
        total_profit = sum(wins)
        total_loss = sum(losses)

        profit_factor = (
            total_profit / abs(total_loss) if total_loss != 0 else float("inf")
        )

        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0

        avg_trade = net_profit / executed_count
        avg_win = total_profit / win_count if win_count > 0 else 0.0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

        return {
            "total_trades": total_count,
            "executed_trades": executed_count,
            "skipped_trades": skipped_count,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "profit_factor": profit_factor,
            "max_win": max_win,
            "max_loss": max_loss,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    def _generate_empty_results(self) -> Dict[str, Any]:
        """Returns default empty results structure."""
        return {
            "initial_equity": self.initial_equity,
            "ending_equity": self.initial_equity,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "monthly_returns": {},
            "trade_statistics": {
                "total_trades": 0,
                "executed_trades": 0,
                "skipped_trades": 0,
                "win_rate": 0.0,
                "net_profit": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "profit_factor": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            },
            "conflict_report": [],
            "conflict_frequency": 0,
            "conflict_rate": 0.0,
            "skipped_profit": 0.0,
            "avoided_loss": 0.0,
            "capital_efficiency": 0.0,
            "margin_efficiency": 0.0,
            "time_weighted_capital_efficiency": 0.0,
            "time_weighted_margin_efficiency": 0.0,
            "average_margin_usage": 0.0,
            "time_weighted_average_margin_usage": 0.0,
            "peak_margin_usage": 0.0,
            "average_concurrent_positions": 0.0,
            "time_weighted_average_concurrent_positions": 0.0,
            "maximum_concurrent_positions": 0,
        }


# ==============================================================================
# Usage Example
# ==============================================================================

def run_example() -> None:
    """Runnable usage demonstration of the SharedCapitalEngine with Capital Conflict Analysis."""
    print("--- Running Shared Capital Engine Example ---")
    
    # Create mock trades including a simultaneous request conflict
    base_time = datetime(2026, 1, 1, 10, 0, 0)
    mock_trades = [
        # Trade 1: Strat A requests capital at base_time. Profit = +100
        Trade(
            strategy_name="Strat_A",
            trade_id=1,
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=2),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=8.0,
            position_value=800.0,  # Required margin = 800 (at 1x leverage)
            commission=1.0,
            profit=79.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=2)
        ),
        # Trade 2: Strat B requests capital simultaneously at base_time. Profit = +50 (would be winner)
        # But required margin is 500. Remaining available margin is 1000 - 800 = 200.
        # Should be skipped! (Conflict loser)
        Trade(
            strategy_name="Strat_B",
            trade_id=2,
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=3),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=5.0,
            position_value=500.0,  # Required margin = 500
            commission=1.0,
            profit=49.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=3)
        ),
        # Trade 3: Strat C requests capital simultaneously at base_time. Profit = -60 (would be loss)
        # Required margin is 300. Remaining available margin is 200.
        # Should be skipped! (Conflict loser - avoided loss)
        Trade(
            strategy_name="Strat_C",
            trade_id=3,
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            side="Long",
            entry_price=100.0,
            exit_price=80.0,
            contracts=3.0,
            position_value=300.0,  # Required margin = 300
            commission=1.0,
            profit=-61.0,
            profit_percent=-20.0,
            holding_time=timedelta(hours=1)
        ),
    ]

    # Run engine at 1x leverage to easily trigger conflict
    engine = SharedCapitalEngine(initial_equity=1000.0, leverage=1.0)
    results = engine.run(mock_trades)

    print(f"Initial Equity: ${results['initial_equity']:.2f}")
    print(f"Ending Equity: ${results['ending_equity']:.2f}")
    print(f"CAGR: {results['cagr'] * 100:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print("\nTrade Statistics:")
    for k, v in results["trade_statistics"].items():
        print(f"  {k}: {v}")
    
    print("\nCapital Conflict Report:")
    for log in results["conflict_report"]:
        print(
            f"  Time: {log['conflict_time']}, Strategy: {log['strategy']}, "
            f"Required Margin: ${log['required_margin']:.2f}, Avail Margin: ${log['available_margin']:.2f}, "
            f"Winner: {log['winner']}, Loser: {log['loser']}"
        )
              
    print("\nCapital Conflict Metrics:")
    print(f"  Conflict Frequency: {results['conflict_frequency']}")
    print(f"  Conflict Rate: {results['conflict_rate'] * 100:.2f}%")
    print(f"  Skipped Profit: ${results['skipped_profit']:.2f}")
    print(f"  Avoided Loss: ${results['avoided_loss']:.2f}")
    print(f"  Capital Efficiency: {results['capital_efficiency']:.4f}")
    print(f"  Average Margin Usage: ${results['average_margin_usage']:.2f}")
    print(f"  Peak Margin Usage: ${results['peak_margin_usage']:.2f}")
    print(f"  Average Concurrent Positions: {results['average_concurrent_positions']:.2f}")
    print(f"  Maximum Concurrent Positions: {results['maximum_concurrent_positions']}")


if __name__ == "__main__":
    run_example()
