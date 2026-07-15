"""
Unit Tests for the Portfolio Risk Engine.
"""

from datetime import datetime, timedelta
import pytest
from portfolio.risk import RiskParameters, RiskEngine, RiskReport


def test_fixed_sizing():
    """Tests fixed quantity and fixed capital sizing modes."""
    # Fixed Quantity
    params = RiskParameters(sizing_mode="fixed_qty", fixed_qty=5.5)
    engine = RiskEngine(params)
    assert engine.calculate_position_size(equity=10000, entry_price=100) == 5.5

    # Fixed Capital
    params.sizing_mode = "fixed_capital"
    params.fixed_capital = 2000.0
    assert engine.calculate_position_size(equity=10000, entry_price=50) == 40.0
    assert engine.calculate_position_size(equity=10000, entry_price=200) == 10.0


def test_risk_per_trade_sizing():
    """Tests sizing based on risk per trade percent and stop loss."""
    params = RiskParameters(sizing_mode="risk_per_trade", risk_pct=0.01)  # 1%
    engine = RiskEngine(params)

    # 1. Using stop loss distance
    # Equity = 10000 -> 1% risk = 100 USDT.
    # Stop loss distance = 2.0 USDT.
    # Expected size = 100 / 2.0 = 50.0 contracts.
    size = engine.calculate_position_size(equity=10000, entry_price=100, stop_loss_dist=2.0)
    assert size == 50.0

    # 2. Using stop loss percentage
    # Equity = 10000 -> 1% risk = 100 USDT.
    # Stop loss pct = 5% (0.05). Entry = 100 -> stop distance = 5.0 USDT.
    # Expected size = 100 / 5.0 = 20.0 contracts.
    size2 = engine.calculate_position_size(equity=10000, entry_price=100, stop_loss_pct=0.05)
    assert size2 == 20.0

    # Test invalid parameters
    assert engine.calculate_position_size(equity=10000, entry_price=100) == 0.0


def test_kelly_sizing():
    """Tests fractional Kelly sizing calculations."""
    params = RiskParameters(sizing_mode="kelly", kelly_fraction=0.5)  # Half-Kelly
    engine = RiskEngine(params)

    # Win rate = 60%, win/loss ratio = 2.0.
    # Kelly = 0.60 - (1 - 0.60) / 2.0 = 0.60 - 0.40 / 2.0 = 0.60 - 0.20 = 0.40.
    # Half-Kelly = 0.20.
    # Size in contracts = (10000 * 0.20) / 100 = 20.0
    size = engine.calculate_position_size(
        equity=10000, entry_price=100, win_rate=0.60, win_loss_ratio=2.0
    )
    assert pytest.approx(size) == 20.0

    # Test negative Kelly size gets bounded to 0.0
    # Win rate = 30%, win/loss ratio = 1.0 -> Kelly = 0.3 - 0.7 = -0.4.
    size_neg = engine.calculate_position_size(
        equity=10000, entry_price=100, win_rate=0.30, win_loss_ratio=1.0
    )
    assert size_neg == 0.0

    # Missing win rate/ratio
    assert engine.calculate_position_size(equity=10000, entry_price=100) == 0.0


def test_volatility_target_sizing():
    """Tests position sizing based on portfolio volatility targeting."""
    params = RiskParameters(sizing_mode="vol_target", target_vol_ann=0.10)  # 10%
    engine = RiskEngine(params)

    # Equity = 10000, asset vol = 20% (0.20), entry = 100.
    # Target weight = 0.10 / 0.20 = 0.5 (50%).
    # Position Value = 10000 * 0.5 = 5000 USDT.
    # Size in contracts = 5000 / 100 = 50.0.
    size = engine.calculate_position_size(
        equity=10000, entry_price=100, asset_vol_ann=0.20
    )
    assert size == 50.0

    # Missing asset vol
    assert engine.calculate_position_size(equity=10000, entry_price=100) == 0.0


def test_atr_sizing():
    """Tests ATR-based position sizing."""
    params = RiskParameters(sizing_mode="atr", risk_pct=0.02, atr_multiplier=2.5)  # 2% risk
    engine = RiskEngine(params)

    # Equity = 10000 -> 2% risk = 200 USDT.
    # ATR = 4.0, stop distance = 4.0 * 2.5 = 10.0 USDT.
    # Expected size = 200 / 10.0 = 20.0 contracts.
    size = engine.calculate_position_size(equity=10000, entry_price=100, atr=4.0)
    assert size == 20.0

    # Missing/invalid ATR
    assert engine.calculate_position_size(equity=10000, entry_price=100) == 0.0


def test_maximum_concurrent_positions_limit():
    """Tests concurrent positions ceiling rejection."""
    params = RiskParameters(max_concurrent_positions=3)
    engine = RiskEngine(params)
    t = datetime(2026, 7, 13, 12, 0, 0)

    # Below limit (2 open positions)
    assert engine.validate_trade_entry(
        ticker="BTC",
        quantity=1.0,
        price=100,
        equity=10000,
        active_positions_count=2,
        current_margin_used=1000,
        current_portfolio_risk=100,
        timestamp=t,
    ) is True

    # At/Above limit (3 open positions)
    assert engine.validate_trade_entry(
        ticker="ETH",
        quantity=1.0,
        price=100,
        equity=10000,
        active_positions_count=3,
        current_margin_used=1000,
        current_portfolio_risk=100,
        timestamp=t,
    ) is False
    assert engine.block_reasons["max_concurrent_positions"] == 1


def test_maximum_margin_usage_limit():
    """Tests margin usage fraction limits constraint."""
    params = RiskParameters(max_margin_usage_pct=0.75, leverage=5.0)  # 75% margin limit, 5x leverage
    engine = RiskEngine(params)
    t = datetime(2026, 7, 13, 12, 0, 0)

    # Equity = 10000. Limit margin used = 7500.
    # Current margin used = 6000.
    # Proposed trade: qty = 20, price = 200 -> value = 4000. Required margin = 4000 / 5 = 800.
    # Total margin if filled = 6800 / 10000 = 68% -> OK.
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=20,
        price=200,
        equity=10000,
        active_positions_count=1,
        current_margin_used=6000,
        current_portfolio_risk=100,
        timestamp=t,
    ) is True

    # Proposed trade: qty = 200, price = 200 -> value = 40000. Required margin = 8000.
    # Total margin if filled = 6000 + 8000 = 14000 / 10000 = 140% -> REJECT.
    assert engine.validate_trade_entry(
        ticker="TSLA",
        quantity=200,
        price=200,
        equity=10000,
        active_positions_count=1,
        current_margin_used=6000,
        current_portfolio_risk=100,
        timestamp=t,
    ) is False
    assert engine.block_reasons["max_margin_usage"] == 1


def test_maximum_drawdown_stop():
    """Tests drawdown stop trigger which halts all future entries."""
    params = RiskParameters(max_drawdown_pct=0.15)  # 15% drawdown threshold
    engine = RiskEngine(params)
    t = datetime(2026, 7, 13, 12, 0, 0)

    # Equity is fine at first
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=1.0,
        price=100,
        equity=10000,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t,
    ) is True

    # Equity drops to 8000 (20% drawdown) -> triggers halt
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=1.0,
        price=100,
        equity=8000,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t,
    ) is False
    assert engine.trading_halted is True
    assert engine.block_reasons["max_drawdown"] == 1

    # Future entries blocked automatically
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=1.0,
        price=100,
        equity=10000,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t,
    ) is False
    assert engine.block_reasons["max_drawdown"] == 2


def test_maximum_portfolio_risk_limit():
    """Tests limit on total portfolio dollar risk."""
    params = RiskParameters(max_portfolio_risk_pct=0.05)  # 5% max risk
    engine = RiskEngine(params)
    t = datetime(2026, 7, 13, 12, 0, 0)

    # Equity = 10000. Limit risk = 500 USDT.
    # Current risk = 300 USDT.
    # Proposed trade: qty = 10, stop_loss_dist = 15.0 -> risk = 150.
    # Total risk = 450 (4.5% of equity) -> OK.
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=10,
        price=100,
        equity=10000,
        active_positions_count=1,
        current_margin_used=1000,
        current_portfolio_risk=300.0,
        timestamp=t,
        stop_loss_dist=15.0,
    ) is True

    # Proposed trade: risk = 250 -> Total risk = 550 (5.5%) -> REJECT
    assert engine.validate_trade_entry(
        ticker="TSLA",
        quantity=10,
        price=200,
        equity=10000,
        active_positions_count=1,
        current_margin_used=1000,
        current_portfolio_risk=300.0,
        timestamp=t,
        stop_loss_dist=25.0,
    ) is False
    assert engine.block_reasons["max_portfolio_risk"] == 1


def test_maximum_daily_loss_limit():
    """Tests tracking of daily realized pnl and daily starting equity checks."""
    params = RiskParameters(max_daily_loss_pct=0.02)  # 2% max daily loss limit
    engine = RiskEngine(params)
    t1 = datetime(2026, 7, 13, 9, 0, 0)
    t2 = datetime(2026, 7, 13, 14, 0, 0)
    t3 = datetime(2026, 7, 14, 9, 0, 0)  # Next day

    # Starting day 1: Equity = 10000
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=1.0,
        price=100,
        equity=10000,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t1,
    ) is True

    # Intraday loss of 150 (1.5%) -> still allowed
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=1.0,
        price=100,
        equity=9850,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t2,
    ) is True

    # Intraday loss of 250 (2.5%) -> blocked
    assert engine.validate_trade_entry(
        ticker="TSLA",
        quantity=1.0,
        price=200,
        equity=9750,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t2,
    ) is False
    assert engine.block_reasons["max_daily_loss"] == 1

    # Next day: starting equity reset to 9750. Loss of 0 -> allowed!
    assert engine.validate_trade_entry(
        ticker="AAPL",
        quantity=1.0,
        price=100,
        equity=9750,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t3,
    ) is True


def test_risk_report():
    """Tests risk report statistics generation."""
    params = RiskParameters(max_concurrent_positions=1)
    engine = RiskEngine(params)
    t = datetime(2026, 7, 13, 12, 0, 0)

    engine.validate_trade_entry(
        ticker="AAPL",
        quantity=1.0,
        price=100,
        equity=10000,
        active_positions_count=0,
        current_margin_used=0.0,
        current_portfolio_risk=0.0,
        timestamp=t,
    )
    # Blocked entry
    engine.validate_trade_entry(
        ticker="TSLA",
        quantity=1.0,
        price=100,
        equity=10000,
        active_positions_count=1,
        current_margin_used=100.0,
        current_portfolio_risk=0.0,
        timestamp=t,
    )

    report = engine.generate_report(initial_equity=10000.0, ending_equity=10200.0)
    assert report.total_trades_evaluated == 2
    assert report.trades_blocked == 1
    assert report.block_reasons["max_concurrent_positions"] == 1
    assert report.initial_equity == 10000.0
    assert report.ending_equity == 10200.0
    assert report.peak_equity == 10000.0
