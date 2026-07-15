"""
Unit and integration tests for the Shared Capital Simulation Engine.
"""

from datetime import datetime, timedelta
import os
import pytest

from portfolio.importer import Trade, parse_tradingview_file, import_tradingview_files
from portfolio.shared_engine import (
    SharedCapitalEngine,
    MarginRequestEvent,
    MarginReleaseEvent,
)
from portfolio.queue import EventQueue

# Root directory of the sample files
SAMPLE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "tradingview-xlsx-export-samples"
    )
)


def test_shared_event_sorting():
    """Verify that MarginReleaseEvent (exits) sorts before MarginRequestEvent (entries) at same timestamp."""
    eq = EventQueue()
    t = datetime(2026, 7, 1, 12, 0, 0)
    
    mock_trade = Trade(
        strategy_name="Strat1",
        trade_id=1,
        entry_time=t,
        exit_time=t,
        side="Long",
        entry_price=100.0,
        exit_price=105.0,
        contracts=10.0,
        position_value=1000.0,
        commission=0.0,
        profit=50.0,
        profit_percent=5.0,
        holding_time=timedelta(0)
    )

    req = MarginRequestEvent(
        timestamp=t,
        trade_id=1,
        position_value=1000.0,
        profit=50.0,
        side="Long",
        strategy_name="Strat1",
        trade=mock_trade,
    )
    rel = MarginReleaseEvent(
        timestamp=t,
        trade_id=1,
        position_value=1000.0,
        profit=50.0,
        side="Long",
        strategy_name="Strat1",
        trade=mock_trade,
    )

    # Push Request first then Release
    eq.push(req)
    eq.push(rel)

    # Pop should return Release (exit) first because alphabetically 'MarginReleaseEvent' < 'MarginRequestEvent'
    first = eq.pop()
    second = eq.pop()

    assert isinstance(first, MarginReleaseEvent)
    assert isinstance(second, MarginRequestEvent)


def test_shared_capital_margin_execution():
    """Test engine margin validation and skipping of trades when available margin is insufficient."""
    engine = SharedCapitalEngine(initial_equity=1000.0, leverage=10.0)
    
    # 10x leverage means 1000 USDT initial equity can support up to 10,000 USDT total position value.
    t = datetime(2026, 7, 1, 12, 0, 0)
    trades = [
        # Trade 1: Position Value = 8,000 USDT. Required Margin = 800 USDT. Available Margin = 1000 USDT.
        Trade(
            strategy_name="Strat_A",
            trade_id=1,
            entry_time=t,
            exit_time=t + timedelta(hours=2),
            side="Long",
            entry_price=100.0,
            exit_price=105.0,
            contracts=80.0,
            position_value=8000.0,
            commission=0.0,
            profit=400.0,
            profit_percent=5.0,
            holding_time=timedelta(hours=2)
        ),
        # Entered while Trade 1 is open. Required Margin (500) > Remaining Available Margin (400). Should skip!
        Trade(
            strategy_name="Strat_B",
            trade_id=2,
            entry_time=t + timedelta(hours=1),
            exit_time=t + timedelta(hours=3),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=50.0,
            position_value=5000.0,
            commission=0.0,
            profit=300.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=2)
        ),
    ]

    results = engine.run(trades)

    # Trade 1 should be executed, Trade 2 should be skipped
    stats = results["trade_statistics"]
    assert stats["total_trades"] == 2
    assert stats["executed_trades"] == 1
    assert stats["skipped_trades"] == 1
    assert results["ending_equity"] == 1400.0  # 1000 + 400 profit from Trade 1
    assert results["max_drawdown"] == 0.0      # Only profit, no drawdown


def test_shared_capital_drawdown_and_cagr():
    """Verify drawdown, max drawdown, and CAGR calculations."""
    engine = SharedCapitalEngine(initial_equity=1000.0, leverage=10.0)
    t = datetime(2026, 1, 1, 0, 0, 0)
    
    trades = [
        # First trade makes a profit
        Trade(
            strategy_name="Strat_A",
            trade_id=1,
            entry_time=t,
            exit_time=t + timedelta(days=10),
            side="Long",
            entry_price=100.0,
            exit_price=105.0,
            contracts=10.0,
            position_value=1000.0,
            commission=0.0,
            profit=50.0,
            profit_percent=5.0,
            holding_time=timedelta(days=10)
        ),
        # Second trade makes a loss, creating drawdown
        Trade(
            strategy_name="Strat_B",
            trade_id=2,
            entry_time=t + timedelta(days=15),
            exit_time=t + timedelta(days=25),
            side="Long",
            entry_price=100.0,
            exit_price=90.0,
            contracts=10.0,
            position_value=1000.0,
            commission=0.0,
            profit=-100.0,
            profit_percent=-10.0,
            holding_time=timedelta(days=10)
        ),
        # Third trade recovers some equity
        Trade(
            strategy_name="Strat_C",
            trade_id=3,
            entry_time=t + timedelta(days=30),
            exit_time=t + timedelta(days=365),  # Simulation runs for a full year
            side="Long",
            entry_price=100.0,
            exit_price=115.0,
            contracts=10.0,
            position_value=1000.0,
            commission=0.0,
            profit=150.0,
            profit_percent=15.0,
            holding_time=timedelta(days=335)
        ),
    ]

    results = engine.run(trades)

    # Peak equity was 1050 (after Trade 1).
    # Equity dropped to 950 (after Trade 2).
    # Drawdown = (1050 - 950) / 1050 = 100 / 1050 ≈ 9.52%
    assert abs(results["max_drawdown"] - 9.5238) < 1e-2

    # Ending equity is 1100 (1000 + 50 - 100 + 150)
    assert results["ending_equity"] == 1100.0

    # Test CAGR over 1 year (duration of snapshots is ~365 days, start to end)
    # CAGR = (1100 / 1000) ^ (1 / 1) - 1 = 10%
    assert abs(results["cagr"] - 0.10) < 1e-2


def test_shared_capital_monthly_returns():
    """Verify monthly return grouping and calculation."""
    engine = SharedCapitalEngine(initial_equity=1000.0, leverage=10.0)
    
    trades = [
        # Exits in Jan 2026
        Trade(
            strategy_name="S",
            trade_id=1,
            entry_time=datetime(2026, 1, 1, 10),
            exit_time=datetime(2026, 1, 15, 10),
            side="Long",
            entry_price=100.0,
            exit_price=105.0,
            contracts=10.0,
            position_value=1000.0,
            commission=0.0,
            profit=50.0,
            profit_percent=5.0,
            holding_time=timedelta(days=14)
        ),
        # Exits in Feb 2026
        Trade(
            strategy_name="S",
            trade_id=2,
            entry_time=datetime(2026, 1, 20, 10),
            exit_time=datetime(2026, 2, 10, 10),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=10.0,
            position_value=1000.0,
            commission=0.0,
            profit=100.0,
            profit_percent=10.0,
            holding_time=timedelta(days=21)
        ),
        # Exits in Mar 2026
        Trade(
            strategy_name="S",
            trade_id=3,
            entry_time=datetime(2026, 2, 25, 10),
            exit_time=datetime(2026, 3, 5, 10),
            side="Long",
            entry_price=100.0,
            exit_price=90.0,
            contracts=10.0,
            position_value=1000.0,
            commission=0.0,
            profit=-150.0,
            profit_percent=-15.0,
            holding_time=timedelta(days=8)
        )
    ]

    results = engine.run(trades)
    monthly = results["monthly_returns"]

    # Month 1 (Jan 2026): End Equity = 1050. Return = (1050 - 1000) / 1000 = 5%
    assert abs(monthly["2026-01"] - 5.0) < 1e-5

    # Month 2 (Feb 2026): End Equity = 1150. Return = (1150 - 1050) / 1050 = 9.5238%
    assert abs(monthly["2026-02"] - 9.5238) < 1e-3

    # Month 3 (Mar 2026): End Equity = 1000. Return = (1000 - 1150) / 1150 = -13.043%
    assert abs(monthly["2026-03"] - (-13.0435)) < 1e-3


def test_shared_capital_real_integration():
    """Integration test loading actual exported TradingView reports and simulating shared capital."""
    if not os.path.exists(SAMPLE_DIR):
        pytest.skip(f"Sample folder not found at {SAMPLE_DIR}")
        
    trades = import_tradingview_files(SAMPLE_DIR)
    assert len(trades) > 0
    
    # Instantiate engine and run backtest
    engine = SharedCapitalEngine(initial_equity=1000.0, leverage=10.0)
    results = engine.run(trades)
    
    assert results["initial_equity"] == 1000.0
    assert "ending_equity" in results
    assert len(results["equity_curve"]) > 0
    assert len(results["drawdown_curve"]) > 0
    assert len(results["monthly_returns"]) > 0
    
    stats = results["trade_statistics"]
    assert stats["total_trades"] == len(trades)
    assert stats["executed_trades"] + stats["skipped_trades"] == len(trades)
    assert "win_rate" in stats
    assert "profit_factor" in stats


def test_shared_capital_conflict_analysis():
    """Verify conflict report logging, Winner/Loser flags, and conflict metrics under constraint."""
    engine = SharedCapitalEngine(initial_equity=1000.0, leverage=1.0)
    t = datetime(2026, 7, 1, 12, 0, 0)
    
    trades = [
        Trade(
            strategy_name="Strat_A",
            trade_id=1,
            entry_time=t,
            exit_time=t + timedelta(hours=2),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=8.0,
            position_value=800.0,  # margin = 800
            commission=0.0,
            profit=80.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=2)
        ),
        Trade(
            strategy_name="Strat_B",
            trade_id=2,
            entry_time=t,
            exit_time=t + timedelta(hours=3),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=5.0,
            position_value=500.0,  # margin = 500. Skipped!
            commission=0.0,
            profit=50.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=3)
        ),
        Trade(
            strategy_name="Strat_C",
            trade_id=3,
            entry_time=t,
            exit_time=t + timedelta(hours=1),
            side="Long",
            entry_price=100.0,
            exit_price=80.0,
            contracts=3.0,
            position_value=300.0,  # margin = 300. Skipped! (Avoided loss)
            commission=0.0,
            profit=-60.0,
            profit_percent=-20.0,
            holding_time=timedelta(hours=1)
        ),
    ]
    
    results = engine.run(trades)
    
    assert results["conflict_frequency"] == 1
    assert results["skipped_profit"] == 50.0
    assert results["avoided_loss"] == 60.0
    assert results["peak_margin_usage"] == 800.0
    assert results["maximum_concurrent_positions"] == 1
    
    report = results["conflict_report"]
    assert len(report) == 3
    
    # Check Strat_A log (winner)
    log_a = next(log for log in report if log["strategy"] == "Strat_A")
    assert log_a["winner"] is True
    assert log_a["loser"] is False
    assert log_a["required_margin"] == 800.0
    assert log_a["available_margin"] == 1000.0
    assert log_a["skipped_trade"] is None
    
    # Check Strat_B log (loser)
    log_b = next(log for log in report if log["strategy"] == "Strat_B")
    assert log_b["winner"] is False
    assert log_b["loser"] is True
    assert log_b["required_margin"] == 500.0
    assert log_b["available_margin"] == 1000.0
    assert log_b["skipped_trade"].profit == 50.0
    
    # Check Strat_C log (loser)
    log_c = next(log for log in report if log["strategy"] == "Strat_C")
    assert log_c["winner"] is False
    assert log_c["loser"] is True
    assert log_c["required_margin"] == 300.0
    assert log_c["available_margin"] == 1000.0
    assert log_c["skipped_trade"].profit == -60.0


def test_no_conflict_for_same_strategy():
    """Verify that multiple simultaneous requests from the same strategy do not trigger a conflict."""
    engine = SharedCapitalEngine(initial_equity=1000.0, leverage=1.0)
    t = datetime(2026, 7, 1, 12, 0, 0)
    
    trades = [
        Trade(
            strategy_name="Strat_A",
            trade_id=1,
            entry_time=t,
            exit_time=t + timedelta(hours=2),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=8.0,
            position_value=800.0,
            commission=0.0,
            profit=80.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=2)
        ),
        # Same strategy, Strat_A
        Trade(
            strategy_name="Strat_A",
            trade_id=2,
            entry_time=t,
            exit_time=t + timedelta(hours=3),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=5.0,
            position_value=500.0,
            commission=0.0,
            profit=50.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=3)
        ),
    ]
    
    results = engine.run(trades)
    
    # No multi-strategy conflict since both requests are from the same strategy (Strat_A)
    assert results["conflict_frequency"] == 0
    assert len(results["conflict_report"]) == 0

def test_dynamic_scaling():
    """Verify that trades are dynamically scaled based on strategy_weights and initial_capital."""
    engine = SharedCapitalEngine(
        initial_equity=1000.0, 
        leverage=1.0, 
        strategy_weights={"Strat_A": 0.5}
    )
    t = datetime(2026, 7, 1, 12, 0, 0)
    
    trades = [
        Trade(
            strategy_name="Strat_A",
            trade_id=1,
            entry_time=t,
            exit_time=t + timedelta(hours=2),
            side="Long",
            entry_price=100.0,
            exit_price=110.0,
            contracts=10.0,
            position_value=1000.0,
            commission=1.0,
            profit=100.0,
            profit_percent=10.0,
            holding_time=timedelta(hours=2),
            initial_capital=10000.0
        ),
    ]
    
    results = engine.run(trades)
    
    scaled_trades = results.get("scaled_trades")
    assert scaled_trades is not None
    assert len(scaled_trades) == 1
    
    # Target capital for Strat_A: 1000 (equity) * 0.5 (weight) = 500
    # Strat_A initial capital: 10000
    # Scale factor: 500 / 10000 = 0.05
    
    scaled_trade = scaled_trades[0]
    assert pytest.approx(scaled_trade.contracts) == 0.5         # 10.0 * 0.05
    assert pytest.approx(scaled_trade.position_value) == 50.0   # 1000.0 * 0.05
    assert pytest.approx(scaled_trade.profit) == 5.0            # 100.0 * 0.05
    assert pytest.approx(scaled_trade.commission) == 0.05       # 1.0 * 0.05
    
    # Check that realized equity was also correctly scaled
    assert pytest.approx(results["ending_equity"]) == 1005.0    # 1000.0 + 5.0
    
    # Check leverage returned in results
    assert results["leverage"] == 1.0

