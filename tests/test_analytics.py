"""
Unit tests for the Portfolio Analytics module.
"""

from datetime import datetime, timedelta
import pytest
import matplotlib.pyplot as plt

from portfolio.importer import Trade
from portfolio.analytics import PortfolioAnalytics


def test_trade_metrics_calculation():
    """Verify that trade-by-trade metrics are correctly calculated."""
    t = datetime(2026, 1, 1, 12, 0, 0)
    mock_trades = [
        Trade(
            strategy_name="S1", trade_id=1, entry_time=t, exit_time=t + timedelta(hours=1),
            side="Long", entry_price=10.0, exit_price=12.0, contracts=5.0, position_value=50.0,
            commission=0.0, profit=10.0, profit_percent=20.0, holding_time=timedelta(hours=1)
        ),
        Trade(
            strategy_name="S1", trade_id=2, entry_time=t + timedelta(days=1), exit_time=t + timedelta(days=1, hours=1),
            side="Long", entry_price=10.0, exit_price=8.0, contracts=5.0, position_value=50.0,
            commission=0.0, profit=-10.0, profit_percent=-20.0, holding_time=timedelta(hours=1)
        ),
        Trade(
            strategy_name="S1", trade_id=3, entry_time=t + timedelta(days=2), exit_time=t + timedelta(days=2, hours=1),
            side="Long", entry_price=10.0, exit_price=15.0, contracts=2.0, position_value=20.0,
            commission=0.0, profit=10.0, profit_percent=50.0, holding_time=timedelta(hours=1)
        ),
        Trade(
            strategy_name="S1", trade_id=4, entry_time=t + timedelta(days=3), exit_time=t + timedelta(days=3, hours=1),
            side="Long", entry_price=10.0, exit_price=11.0, contracts=5.0, position_value=50.0,
            commission=0.0, profit=5.0, profit_percent=10.0, holding_time=timedelta(hours=1)
        )
    ]

    analytics = PortfolioAnalytics(trades=mock_trades)
    metrics = analytics.calculate_trade_metrics()

    # Total trades: 4. Wins: 3 (10, 10, 5), Losses: 1 (-10).
    assert metrics["win_rate"] == 75.0
    assert metrics["avg_win"] == 25 / 3  # (10 + 10 + 5) / 3 ≈ 8.33
    assert metrics["avg_loss"] == -10.0
    assert metrics["largest_win"] == 10.0
    assert metrics["largest_loss"] == -10.0
    assert metrics["profit_factor"] == 2.5  # 25 / 10 = 2.5
    # Expectancy = (0.75 * 8.333) + (0.25 * -10) = 6.25 - 2.5 = 3.75
    assert pytest.approx(metrics["expectancy"]) == 3.75


def test_consecutive_streaks():
    """Verify maximum consecutive win/loss streaks."""
    t = datetime(2026, 1, 1)
    # Win, Loss, Win, Win, Win, Loss, Loss
    mock_trades = [
        Trade(
            strategy_name="S1", trade_id=i, entry_time=t + timedelta(days=i), exit_time=t + timedelta(days=i),
            side="Long", entry_price=10.0, exit_price=12.0, contracts=1.0, position_value=10.0,
            commission=0.0, profit=prof, profit_percent=10.0, holding_time=timedelta(0)
        )
        for i, prof in enumerate([10.0, -5.0, 8.0, 12.0, 5.0, -10.0, -2.0])
    ]

    analytics = PortfolioAnalytics(trades=mock_trades)
    metrics = analytics.calculate_trade_metrics()

    assert metrics["max_consecutive_wins"] == 3
    assert metrics["max_consecutive_losses"] == 2


def test_equity_metrics():
    """Verify CAGR, drawdowns, Sharpe, Sortino, MAR, Ulcer Index."""
    t = datetime(2026, 1, 1)
    
    # 1 year of daily returns
    equity_curve = []
    current_eq = 1000.0
    equity_curve.append((t, current_eq))
    
    # Simple steady returns
    for i in range(1, 366):
        current_eq += 2.0  # Constant profit
        equity_curve.append((t + timedelta(days=i), current_eq))

    analytics = PortfolioAnalytics(equity_curve=equity_curve, initial_equity=1000.0)
    metrics = analytics.calculate_all_metrics()

    # CAGR: (1730 / 1000)^(1/0.996) - 1 ≈ 73%
    assert metrics["cagr"] > 0.70
    assert metrics["max_drawdown_pct"] == 0.0  # Steady increase, no drawdown
    assert metrics["ulcer_index"] == 0.0
    assert metrics["recovery_factor"] == float("inf")
    assert metrics["calmar"] == float("inf")
    
    # Now let's inject a drawdown
    equity_curve_dd = []
    eq = 1000.0
    for i in range(100):
        eq += 10.0
        equity_curve_dd.append((t + timedelta(days=i), eq))
    # Peak at 2000. Drop to 1600. Max DD = 20%
    for i in range(100, 110):
        eq -= 40.0
        equity_curve_dd.append((t + timedelta(days=i), eq))
    for i in range(110, 200):
        eq += 10.0
        equity_curve_dd.append((t + timedelta(days=i), eq))
        
    analytics_dd = PortfolioAnalytics(equity_curve=equity_curve_dd, initial_equity=1000.0)
    
    assert pytest.approx(analytics_dd.calculate_max_drawdown_percent(), abs=1e-3) == 0.20
    assert analytics_dd.calculate_max_drawdown_cash() == 400.0
    assert analytics_dd.calculate_ulcer_index() > 0.0
    assert analytics_dd.calculate_recovery_factor() > 0.0


def test_empty_and_single_elements():
    """Verify graceful handling of empty or single element input."""
    # Empty
    analytics = PortfolioAnalytics()
    metrics = analytics.calculate_all_metrics()
    assert metrics["sharpe"] == 0.0
    assert metrics["sortino"] == 0.0
    assert metrics["cagr"] == 0.0
    assert metrics["max_drawdown_pct"] == 0.0
    assert metrics["recovery_factor"] == 0.0

    # Single item
    t = datetime(2026, 1, 1)
    analytics_single = PortfolioAnalytics(equity_curve=[(t, 1000.0)])
    metrics_single = analytics_single.calculate_all_metrics()
    assert metrics_single["sharpe"] == 0.0
    assert metrics_single["cagr"] == 0.0
    assert metrics_single["max_drawdown_pct"] == 0.0


def test_visualizations():
    """Verify that all plotting functions execute and return figures."""
    t = datetime(2026, 1, 1)
    equity_curve = [
        (t + timedelta(days=i), 1000.0 + (i * 5.0) - (10.0 if i % 10 == 0 else 0.0))
        for i in range(50)
    ]
    
    analytics = PortfolioAnalytics(equity_curve=equity_curve, initial_equity=1000.0)
    
    # Check return type of individual plots
    fig1 = analytics.plot_equity_curve()
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = analytics.plot_drawdown_curve()
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    fig3 = analytics.plot_monthly_return_heatmap()
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    fig4 = analytics.plot_yearly_returns()
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)

    fig5 = analytics.plot_rolling_returns()
    assert isinstance(fig5, plt.Figure)
    plt.close(fig5)

    fig6 = analytics.plot_rolling_drawdown()
    assert isinstance(fig6, plt.Figure)
    plt.close(fig6)

    fig7 = analytics.plot_return_distribution()
    assert isinstance(fig7, plt.Figure)
    plt.close(fig7)

    fig8 = analytics.plot_histogram()
    assert isinstance(fig8, plt.Figure)
    plt.close(fig8)

    # Check composite dashboard
    fig_dash = analytics.generate_dashboard()
    assert isinstance(fig_dash, plt.Figure)
    plt.close(fig_dash)
