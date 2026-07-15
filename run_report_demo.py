"""
Runnable demonstration of the Portfolio Report Generator.

This script simulates a portfolio backtest with capital allocation conflicts and
limit triggers, then compiles a professional PDF, Excel workbook, and CSV segment reports.
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from portfolio.importer import Trade
from portfolio.report import PortfolioReportGenerator
from portfolio.risk import RiskReport, RiskParameters


def run_demo() -> None:
    print("==================================================")
    print("        PORTFOLIO REPORT GENERATOR DEMO           ")
    print("==================================================")

    # 1. Create a synthetic daily equity curve over 18 months
    base_time = datetime(2025, 1, 1)
    np.random.seed(88)
    
    n_days = 540
    daily_pct = np.random.normal(loc=0.0007, scale=0.011, size=n_days)
    
    # Inject major market stress (drawdowns)
    daily_pct[100:130] -= 0.009  # 30-day drawdown
    daily_pct[320:350] -= 0.008  # another drawdown
    
    equity = [100000.0]
    for p in daily_pct:
        next_eq = max(1000.0, equity[-1] * (1.0 + p))
        equity.append(next_eq)
        
    equity_curve = [
        (base_time + timedelta(days=i), eq) for i, eq in enumerate(equity)
    ]
    
    # Compute drawdown curve
    peaks = pd.Series([e[1] for e in equity_curve]).cummax()
    drawdown_curve = []
    for i, (dt, eq) in enumerate(equity_curve):
        dd = (peaks[i] - eq) / peaks[i] * 100.0
        drawdown_curve.append((dt, dd))

    # 2. Create realistic synthetic trades
    trades = []
    strategies = ["AlphaTrend", "BreakoutBot", "MeanReversion"]
    
    current_time = base_time
    for idx in range(1, 121):
        strat = np.random.choice(strategies)
        is_win = np.random.rand() < 0.54  # 54% win rate
        val = np.random.exponential(scale=1800.0)
        profit = val if is_win else -val * 0.85
        
        entry = current_time + timedelta(days=np.random.randint(1, 4))
        exit_time = entry + timedelta(days=np.random.randint(1, 5))
        
        trades.append(
            Trade(
                strategy_name=strat,
                trade_id=idx,
                entry_time=entry,
                exit_time=exit_time,
                side=np.random.choice(["Long", "Short"]),
                entry_price=100.0,
                exit_price=100.0 + (profit / 100.0),
                contracts=10.0,
                position_value=10000.0,
                commission=8.5,
                profit=profit - 17.0,  # includes round-turn commission
                profit_percent=(profit / 10000.0) * 100.0,
                holding_time=(exit_time - entry)
            )
        )
        current_time = exit_time

    # 3. Create monthly returns matrix
    monthly_returns = {}
    eq_series = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
    monthly_snapshots = eq_series.resample("ME").last()
    
    prev_eq = 100000.0
    for dt, eq in monthly_snapshots.items():
        key = f"{dt.year}-{dt.month:02d}"
        ret = (eq - prev_eq) / prev_eq * 100.0
        monthly_returns[key] = ret
        prev_eq = eq

    # 4. Create conflict logs
    conflict_logs = []
    conflict_dates = [base_time + timedelta(days=50), base_time + timedelta(days=120),
                      base_time + timedelta(days=220), base_time + timedelta(days=340),
                      base_time + timedelta(days=480)]
                      
    for c_date in conflict_dates:
        strat_win = np.random.choice(strategies)
        strat_lose = np.random.choice([s for s in strategies if s != strat_win])
        
        # Winner Log
        conflict_logs.append({
            "conflict_time": c_date,
            "strategy": strat_win,
            "trade_id": f"{strat_win}_T_conf",
            "required_margin": 15000.0,
            "available_margin": 22000.0,
            "winner": True,
            "loser": False,
            "skipped_trade": None
        })
        
        # Loser Log (Skipped Trade)
        skipped_trade_profit = np.random.uniform(-4000, 6000)
        conflict_logs.append({
            "conflict_time": c_date,
            "strategy": strat_lose,
            "trade_id": f"{strat_lose}_T_conf",
            "required_margin": 18000.0,
            "available_margin": 22000.0,
            "winner": False,
            "loser": True,
            "skipped_trade": Trade(
                strategy_name=strat_lose, trade_id=999, entry_time=c_date, exit_time=c_date + timedelta(days=2),
                side="Long", entry_price=10.0, exit_price=11.0, contracts=100.0, position_value=18000.0,
                commission=10.0, profit=skipped_trade_profit, profit_percent=skipped_trade_profit/180.0,
                holding_time=timedelta(days=2)
            )
        })

    # Summary metrics calculation
    skipped_profit = sum(log["skipped_trade"].profit for log in conflict_logs if log["loser"] and log["skipped_trade"].profit > 0)
    avoided_loss = sum(abs(log["skipped_trade"].profit) for log in conflict_logs if log["loser"] and log["skipped_trade"].profit < 0)

    # 5. Compile final results dict
    results = {
        "initial_equity": 100000.0,
        "ending_equity": equity[-1],
        "cagr": 0.185,
        "max_drawdown": 12.35,
        "max_drawdown_cash": 14250.0,
        "sharpe": 1.45,
        "sortino": 1.82,
        "calmar": 1.50,
        "ulcer_index": 3.82,
        "recovery_factor": 4.15,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "monthly_returns": monthly_returns,
        "trade_statistics": {
            "total_trades": len(trades),
            "executed_trades": len(trades) - 5,
            "skipped_trades": 5,
            "win_rate": 53.8,
            "net_profit": equity[-1] - 100000.0,
            "total_profit": 143000.0,
            "total_loss": -84000.0,
            "profit_factor": 1.70,
            "max_win": 8200.0,
            "max_loss": -6400.0,
            "avg_trade": 490.0,
            "avg_win": 2100.0,
            "avg_loss": -1650.0,
            "expectancy": 365.0,
            "max_consecutive_wins": 7,
            "max_consecutive_losses": 5,
        },
        "conflict_frequency": 5,
        "conflict_rate": 0.04,
        "skipped_profit": skipped_profit,
        "avoided_loss": avoided_loss,
        "capital_efficiency": 0.22,
        "margin_efficiency": 0.48,
        "time_weighted_capital_efficiency": 0.19,
        "time_weighted_margin_efficiency": 0.43,
        "average_margin_usage": 22000.0,
        "time_weighted_average_margin_usage": 19000.0,
        "peak_margin_usage": 45000.0,
        "average_concurrent_positions": 1.25,
        "maximum_concurrent_positions": 4,
    }

    # 6. Risk Engine Metrics
    risk_report = RiskReport(
        initial_equity=100000.0,
        ending_equity=equity[-1],
        peak_equity=max(equity),
        max_drawdown_pct=0.1235,
        total_trades_evaluated=125,
        trades_blocked=5,
        block_reasons={"max_margin_usage": 3, "max_concurrent_positions": 2},
    )

    risk_params = RiskParameters(
        sizing_mode="risk_per_trade",
        risk_pct=0.01,
        leverage=10.0,
    )

    # 7. Instantiate report generator
    generator = PortfolioReportGenerator(
        backtest_results=results,
        trades=trades,
        conflict_logs=conflict_logs,
        risk_report=risk_report,
        risk_params=risk_params,
        portfolio_name="Alpha Tactical Multi-Strategy Portfolio"
    )

    # Compile Reports
    out_dir = "output_reports"
    os.makedirs(out_dir, exist_ok=True)
    
    pdf_path = os.path.join(out_dir, "portfolio_report.pdf")
    excel_path = os.path.join(out_dir, "portfolio_report.xlsx")
    csv_prefix = os.path.join(out_dir, "portfolio_report")
    
    print("\nGenerating PDF report...")
    generator.generate_pdf(pdf_path)
    
    print("Generating Excel workbook...")
    generator.generate_excel(excel_path)
    
    print("Generating CSV reports...")
    generator.generate_csv(csv_prefix)

    print("\n==================================================")
    print("DEMO RUN SUCCESSFUL! Created report files in:")
    print(f"  PDF:   {pdf_path}")
    print(f"  Excel: {excel_path}")
    print(f"  CSV:   {csv_prefix}_summary.csv, etc.")
    print("==================================================")


if __name__ == "__main__":
    run_demo()
