"""
Unit tests for the Portfolio Optimizer module.
"""

from datetime import datetime, timedelta
import numpy as np
import pytest

from portfolio.importer import Trade
from portfolio.optimizer import PortfolioOptimizer
from portfolio.risk import RiskParameters


def create_mock_trade(
    strategy_name: str,
    trade_id: int,
    entry_days_offset: int,
    exit_days_offset: int,
    profit: float,
    position_value: float = 1000.0,
) -> Trade:
    """Helper to create a mock Trade object."""
    base_time = datetime(2026, 1, 1, 10, 0, 0)
    return Trade(
        strategy_name=strategy_name,
        trade_id=trade_id,
        entry_time=base_time + timedelta(days=entry_days_offset),
        exit_time=base_time + timedelta(days=exit_days_offset),
        side="Long",
        entry_price=100.0,
        exit_price=110.0 if profit > 0 else 90.0,
        contracts=position_value / 100.0,
        position_value=position_value,
        commission=2.0,
        profit=profit,
        profit_percent=profit / position_value,
        holding_time=timedelta(days=exit_days_offset - entry_days_offset),
    )


def test_optimizer_initialization():
    """Verify that the optimizer parses trades and builds the correct matrices."""
    # Strategy 1 trades: profitable
    s1_trades = [
        create_mock_trade("Strat1", 1, 1, 2, 100.0),
        create_mock_trade("Strat1", 2, 5, 6, 150.0),
    ]
    # Strategy 2 trades: profitable
    s2_trades = [
        create_mock_trade("Strat2", 1, 2, 3, 200.0),
        create_mock_trade("Strat2", 2, 7, 8, 50.0),
    ]

    strategies = {"Strat1": s1_trades, "Strat2": s2_trades}
    optimizer = PortfolioOptimizer(
        strategies=strategies,
        initial_equity=10000.0,
        leverage=1.0,
        trading_days=252,
    )

    assert optimizer.strategy_names == ["Strat1", "Strat2"]
    assert optimizer.equity_matrix.shape[1] == 2
    assert len(optimizer.dates) > 0

    # Ensure daily returns matrix is computed and has values
    assert optimizer.returns_matrix.shape[1] == 2


def test_optimization_objectives():
    """Test optimization under different objective functions."""
    s1_trades = [
        create_mock_trade("Strat1", 1, 1, 3, 200.0),
        create_mock_trade("Strat1", 2, 5, 8, 300.0),
    ]
    s2_trades = [
        create_mock_trade("Strat2", 1, 2, 4, -100.0),
        create_mock_trade("Strat2", 2, 6, 9, 400.0),
    ]

    strategies = {"Strat1": s1_trades, "Strat2": s2_trades}
    optimizer = PortfolioOptimizer(
        strategies=strategies,
        initial_equity=10000.0,
        leverage=1.0,
    )

    # Test Sharpe Optimization
    res_sharpe = optimizer.optimize(objective="sharpe", max_leverage=1.0)
    assert "optimal_weights" in res_sharpe
    weights = res_sharpe["optimal_weights"]
    assert len(weights) == 2
    assert sum(weights.values()) <= 1.0001
    assert all(w >= 0.0 for w in weights.values())

    # Test CAGR Optimization
    res_cagr = optimizer.optimize(objective="cagr", max_leverage=1.0)
    assert res_cagr["expected_cagr"] >= 0.0

    # Test Min Max Drawdown Optimization
    res_dd = optimizer.optimize(objective="drawdown", max_leverage=1.0)
    assert res_dd["expected_max_drawdown"] >= 0.0

    # Test Calmar Optimization
    res_calmar = optimizer.optimize(objective="calmar", max_leverage=1.0)
    assert "optimal_weights" in res_calmar

    # Test Sortino Optimization
    res_sortino = optimizer.optimize(objective="sortino", max_leverage=1.0)
    assert "optimal_weights" in res_sortino
    assert "expected_sortino" in res_sortino
    assert res_sortino["expected_sortino"] >= -10.0



def test_optimization_constraints():
    """Test optimizer constraint enforcement."""
    s1_trades = [
        create_mock_trade("Strat1", 1, 1, 2, 500.0),
    ]
    s2_trades = [
        create_mock_trade("Strat2", 1, 1, 2, 300.0),
    ]

    strategies = {"Strat1": s1_trades, "Strat2": s2_trades}
    optimizer = PortfolioOptimizer(
        strategies=strategies,
        initial_equity=10000.0,
        leverage=1.0,
    )

    # Leverage limit: max_leverage = 0.5
    res = optimizer.optimize(objective="sharpe", max_leverage=0.5)
    weights = res["optimal_weights"]
    assert sum(weights.values()) <= 0.5001

    # Weight limits: Strat1 bound to max 0.2
    bounds = {"Strat1": (0.0, 0.2), "Strat2": (0.0, 0.8)}
    res_bounded = optimizer.optimize(objective="sharpe", max_leverage=1.0, weight_bounds=bounds)
    assert res_bounded["optimal_weights"]["Strat1"] <= 0.2001
    assert res_bounded["optimal_weights"]["Strat2"] <= 0.8001

    # Minimum Cash Reserve constraint (e.g. min 9500 cash)
    # Both strategies request 1000 position value.
    # At leverage 1.0, margin is 1000.
    # We constrain min cash to 9500, which means avail_margin >= 9500.
    # Thus, maximum allocation should satisfy this limit.
    res_cash = optimizer.optimize(objective="cagr", max_leverage=1.0, min_cash_reserve=9500.0)
    # Available margin for weights w is: 10000 - w1*1000 - w2*1000 >= 9500 => w1 + w2 <= 0.5
    assert sum(res_cash["optimal_weights"].values()) <= 0.5001


def test_efficient_frontier():
    """Test generation of the efficient frontier."""
    s1_trades = [
        create_mock_trade("Strat1", 1, 1, 3, 200.0),
        create_mock_trade("Strat1", 2, 5, 8, 300.0),
    ]
    s2_trades = [
        create_mock_trade("Strat2", 1, 2, 4, 150.0),
        create_mock_trade("Strat2", 2, 6, 9, 250.0),
    ]

    strategies = {"Strat1": s1_trades, "Strat2": s2_trades}
    optimizer = PortfolioOptimizer(
        strategies=strategies,
        initial_equity=10000.0,
        leverage=1.0,
    )

    frontier = optimizer.generate_efficient_frontier(max_leverage=1.0, points_count=5)
    assert len(frontier) > 0
    for pt in frontier:
        assert "cagr" in pt
        assert "drawdown" in pt
        assert "weights" in pt


def test_simulation_verification():
    """Test verification using event-driven SharedCapitalEngine."""
    s1_trades = [
        create_mock_trade("Strat1", 1, 1, 3, 200.0),
    ]
    s2_trades = [
        create_mock_trade("Strat2", 1, 2, 4, 150.0),
    ]

    strategies = {"Strat1": s1_trades, "Strat2": s2_trades}
    optimizer = PortfolioOptimizer(
        strategies=strategies,
        initial_equity=10000.0,
        leverage=1.0,
    )

    optimal_weights = {"Strat1": 0.6, "Strat2": 0.4}
    sim_res = optimizer.verify_simulation(optimal_weights)

    assert "initial_equity" in sim_res
    assert sim_res["initial_equity"] == 10000.0
    # Expected net profit: 200 * 0.6 + 150 * 0.4 = 120 + 60 = 180 (minus commissions)
    # Single trade Strat1 has entry fee + exit fee, similarly Strat2.
    assert sim_res["ending_equity"] > 10000.0
