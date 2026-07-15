"""
TradingView Portfolio Backtest CLI Entrypoint.

This script parses command-line arguments to simulate a portfolio backtest with
capital allocation conflicts and limit triggers, compiles performance reports
in PDF, Excel, and CSV formats, generates performance charts, and saves them
to the specified output directory.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from portfolio.importer import import_tradingview_files, Trade
from portfolio.risk import RiskParameters, RiskEngine
from portfolio.shared_engine import SharedCapitalEngine
from portfolio.report import PortfolioReportGenerator
from portfolio.optimizer import PortfolioOptimizer

# Setup logger for the CLI script
logger = logging.getLogger("portfolio.cli")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main() -> None:
    """
    Parses command-line arguments, executes the backtest simulation,
    and saves output reports and charts.
    """
    parser = argparse.ArgumentParser(
        description="TradingView Portfolio Backtest CLI Framework"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Initial capital/equity in USDT (default: 1000.0)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=10.0,
        help="Leverage factor (default: 10.0)",
    )

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Directory path containing TradingView Excel reports to import",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_reports",
        help="Directory to save generated reports and charts (default: output_reports)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable portfolio optimization across imported strategies",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="sharpe",
        choices=["cagr", "drawdown", "sharpe", "calmar", "recovery_factor", "cvar"],
        help="Optimization objective (default: sharpe)",
    )
    parser.add_argument(
        "--cvar-iterations",
        type=int,
        default=1000,
        help="Number of Monte Carlo paths for CVaR (default: 1000)",
    )
    parser.add_argument(
        "--cvar-confidence",
        type=float,
        default=0.95,
        help="Confidence level for CVaR (default: 0.95)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Minimum allocation weight per strategy (default: 0.0)",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
        help="Maximum allocation weight per strategy (default: 1.0)",
    )
    parser.add_argument(
        "--min-cash",
        type=float,
        default=0.0,
        help="Minimum cash reserve (as fraction if <= 1.0, else absolute capital) (default: 0.0)",
    )
    parser.add_argument(
        "--max-risk",
        type=float,
        default=None,
        help="Maximum portfolio risk as annualized volatility limit (e.g. 0.15 for 15%)",
    )
    parser.add_argument(
        "--max-concurrent-positions",
        type=int,
        default=None,
        help="Maximum allowed concurrent strategy positions",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=1.0,
        help="Maximum portfolio drawdown fraction before halting trading (default: 1.0 / no halt)",
    )
    parser.add_argument(
        "--disable-risk-limits",
        action="store_true",
        help="Disable the Risk Engine completely to prevent any trade skipping from risk limits",
    )
    parser.add_argument(
        "--stress-test-drawdown",
        action="store_true",
        help="Enable worst-case floating PnL modeling based on Maximum Adverse Excursion (MAE) instead of linear interpolation.",
    )

    args = parser.parse_args()

    # Normalize folder path and check existence
    folder_path = os.path.abspath(args.folder)
    if not os.path.exists(folder_path):
        logger.error(f"Input directory does not exist: {folder_path}")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        logger.error(f"Provided input path is not a directory: {folder_path}")
        sys.exit(1)

    # Resolve output directory
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Importing TradingView reports from: {folder_path}")
    try:
        trades = import_tradingview_files(folder_path)
    except Exception as e:
        logger.error(f"Failed to import TradingView files: {e}")
        sys.exit(1)

    if not trades:
        logger.error(f"No valid TradingView Excel files containing trades found in {folder_path}")
        sys.exit(1)

    logger.info(f"Imported {len(trades)} trades across all strategies.")

    # Configure risk parameters and engine
    if args.disable_risk_limits:
        risk_params = None
        risk_engine = None
    else:
        risk_params = RiskParameters(
            sizing_mode="risk_per_trade",
            leverage=args.leverage,
            max_margin_usage_pct=0.80,
            max_drawdown_pct=args.max_drawdown,
            max_concurrent_positions=args.max_concurrent_positions,
        )
        risk_engine = RiskEngine(risk_params)

    if args.optimize:
        logger.info(f"Optimizing portfolio for objective '{args.objective}'...")
        # Group trades by strategy name
        strategies = {}
        for t in trades:
            strat = t.strategy_name or "Unknown"
            strategies.setdefault(strat, []).append(t)

        weight_bounds = {name: (args.min_weight, args.max_weight) for name in strategies}
        
        try:
            optimizer = PortfolioOptimizer(
                strategies=strategies,
                initial_equity=args.capital,
                leverage=args.leverage,
                stress_test_drawdown=args.stress_test_drawdown,
            )
            
            opt_res = optimizer.optimize(
                objective=args.objective,
                max_leverage=args.leverage,
                max_portfolio_risk=args.max_risk,
                max_concurrent_positions=args.max_concurrent_positions,
                min_cash_reserve=args.min_cash,
                weight_bounds=weight_bounds,
                cvar_iterations=args.cvar_iterations,
                cvar_confidence=args.cvar_confidence,
            )
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            sys.exit(1)

        optimal_weights = opt_res["optimal_weights"]
        print("\n" + "=" * 50)
        print("            PORTFOLIO OPTIMIZATION RESULTS        ")
        print("=" * 50)
        print(f"Objective:                  {args.objective.upper()}")
        print(f"Optimization Status:        {'SUCCESS' if opt_res['success'] else 'FAILED'} ({opt_res['message']})")
        print("-" * 50)
        print("Optimal Strategy Weights:")
        for strat, w in optimal_weights.items():
            print(f"  - {strat:<25}: {w * 100:.2f}% (${w * args.capital:,.2f})")
        print("-" * 50)
        print(f"Expected CAGR:              {opt_res['expected_cagr'] * 100:.2f}%")
        print(f"Expected Max Drawdown:      {opt_res['expected_max_drawdown'] * 100:.2f}%")
        if 'expected_cvar' in opt_res:
            print(f"Expected MC CVaR (95%):     {opt_res['expected_cvar'] * 100:.2f}%")
        print(f"Expected Sharpe:            {opt_res['expected_sharpe']:.2f}")
        print(f"Expected Calmar:            {opt_res['expected_calmar']:.2f}")
        print(f"Expected Recovery Factor:   {opt_res['expected_recovery_factor']:.2f}")
        print("=" * 50 + "\n")

        # Generate and save Efficient Frontier plot
        try:
            frontier = optimizer.generate_efficient_frontier(
                max_leverage=args.leverage,
                min_cash_reserve=args.min_cash,
                weight_bounds=weight_bounds,
                points_count=20
            )
            frontier_path = os.path.join(output_dir, "efficient_frontier.png")
            optimizer.plot_efficient_frontier(frontier, opt_res, save_path=frontier_path)
        except Exception as e:
            logger.error(f"Failed to generate/plot efficient frontier: {e}")

        # Run verification simulation with optimal weights
        logger.info("Running verification simulation with optimal weights...")
        try:
            results = optimizer.verify_simulation(optimal_weights, risk_engine_params=risk_params)
            results["cvar"] = opt_res.get("expected_cvar")
            results["cvar_confidence"] = args.cvar_confidence
            results["cvar_iterations"] = args.cvar_iterations
            results["optimization_objective"] = args.objective
            results["optimal_weights"] = optimal_weights
        except Exception as e:
            logger.error(f"Verification simulation failed: {e}")
            sys.exit(1)

        # Extract dynamically scaled trades from simulation results for reporting
        trades = results.get("scaled_trades", trades)

    else:
        # Initialize shared capital backtest engine
        logger.info(f"Running simulation with Capital: ${args.capital:,.2f}, Leverage: {args.leverage}x")
        engine = SharedCapitalEngine(
            initial_equity=args.capital,
            leverage=args.leverage,
            stress_test_drawdown=args.stress_test_drawdown,
        )
        
        try:
            results = engine.run(trades, risk_engine=risk_engine)
        except Exception as e:
            logger.error(f"Simulation run failed: {e}")
            sys.exit(1)

        # Inject snapshots history for detailed capital usage reports
        results["history"] = engine.history

        # Calculate baseline CVaR
        try:
            import numpy as np
            strategies = {}
            for t in trades:
                strat = t.strategy_name or "Unknown"
                strategies.setdefault(strat, []).append(t)
            
            optimizer = PortfolioOptimizer(
                strategies=strategies,
                initial_equity=args.capital,
                leverage=args.leverage,
                stress_test_drawdown=args.stress_test_drawdown,
            )
            weights = np.ones(len(optimizer.strategy_names))
            cvar = optimizer._calculate_monte_carlo_cvar(weights, iterations=args.cvar_iterations, confidence=args.cvar_confidence)
            results["cvar"] = cvar
            results["cvar_confidence"] = args.cvar_confidence
            results["cvar_iterations"] = args.cvar_iterations
        except Exception as e:
            logger.warning(f"Failed to calculate baseline CVaR: {e}")

    # Update trades list with dynamically scaled versions if available
    trades = results.get("scaled_trades", trades)

    logger.info("Generating reports and charts...")
    generator = PortfolioReportGenerator(
        backtest_results=results,
        trades=trades,
        conflict_logs=results.get("conflict_report", []),
        risk_report=results.get("risk_report"),
        risk_params=results.get("risk_parameters"),
        portfolio_name="TradingView Portfolio Backtest"
    )

    pdf_path = os.path.join(output_dir, "portfolio_report.pdf")
    excel_path = os.path.join(output_dir, "portfolio_report.xlsx")
    csv_prefix = os.path.join(output_dir, "portfolio_report")

    try:
        generator.generate_pdf(pdf_path)
        generator.generate_excel(excel_path)
        generator.generate_csv(csv_prefix)
    except Exception as e:
        logger.error(f"Failed to generate output files: {e}")
        sys.exit(1)

    # Save charts as individual PNG files
    try:
        # Equity & Drawdown
        eq_buf = generator._generate_equity_drawdown_plot()
        with open(os.path.join(output_dir, "equity_drawdown.png"), "wb") as f:
            f.write(eq_buf.getvalue())

        # Monthly Heatmap
        hm_buf = generator._generate_monthly_heatmap_plot()
        with open(os.path.join(output_dir, "monthly_heatmap.png"), "wb") as f:
            f.write(hm_buf.getvalue())

        # Yearly Returns
        yr_buf = generator._generate_yearly_returns_plot()
        with open(os.path.join(output_dir, "yearly_returns.png"), "wb") as f:
            f.write(yr_buf.getvalue())

        # Capital Usage
        cap_buf = generator._generate_capital_usage_plot()
        with open(os.path.join(output_dir, "capital_usage.png"), "wb") as f:
            f.write(cap_buf.getvalue())
    except Exception as e:
        logger.error(f"Failed to generate standalone chart PNG files: {e}")
        sys.exit(1)

    # Compute execution summary metrics
    initial_equity = results.get("initial_equity", args.capital)
    ending_equity = results.get("ending_equity", initial_equity)
    net_profit = ending_equity - initial_equity
    net_profit_pct = (net_profit / initial_equity) * 100.0 if initial_equity > 0 else 0.0
    cagr = results.get("cagr", 0.0) * 100.0
    max_dd = results.get("max_drawdown", 0.0)

    trade_stats = results.get("trade_statistics", {})
    conflict_report = results.get("conflict_report", [])

    print("\n" + "=" * 50)
    print("        PORTFOLIO BACKTEST EXECUTION REPORT       ")
    print("=" * 50)
    print(f"Initial Capital:            ${initial_equity:,.2f}")
    print(f"Ending Capital:             ${ending_equity:,.2f}")
    print(f"Net Profit:                 ${net_profit:+,.2f} ({net_profit_pct:+.2f}%)")
    print(f"CAGR:                       {cagr:.2f}%")
    print(f"Max Drawdown:               {max_dd:.2f}%")
    print("-" * 50)
    print(f"Total Trades Evaluated:     {trade_stats.get('total_trades', len(trades))}")
    print(f"Executed Trades:            {trade_stats.get('executed_trades', 0)}")
    print(f"Skipped Trades (Conflicts): {trade_stats.get('skipped_trades', 0)}")
    print(f"Win Rate:                   {trade_stats.get('win_rate', 0.0):.2f}%")
    print(f"Profit Factor:              {trade_stats.get('profit_factor', 0.0):.2f}")
    print("-" * 50)
    print(f"Conflict Frequency:         {results.get('conflict_frequency', 0)}")
    print(f"Opportunity Cost (Skipped): ${results.get('skipped_profit', 0.0):,.2f}")
    print(f"Avoided Losses (Skipped):   ${results.get('avoided_loss', 0.0):,.2f}")
    
    cap_eff = results.get("capital_efficiency", 0.0)
    if 0.0 < cap_eff <= 1.0:
        cap_eff *= 100.0
    print(f"Capital Efficiency (Avg):   {cap_eff:.2f}%")
    print(f"Peak Margin Usage:          ${results.get('peak_margin_usage', 0.0):,.2f}")
    print("-" * 50)
    print("Outputs Saved:")
    print(f"  - PDF:                    {os.path.join(output_dir, 'portfolio_report.pdf')}")
    print(f"  - Excel:                  {os.path.join(output_dir, 'portfolio_report.xlsx')}")
    print(f"  - CSV Summary:            {os.path.join(output_dir, 'portfolio_report_summary.csv')}")
    print(f"  - CSV Trades:             {os.path.join(output_dir, 'portfolio_report_trades.csv')}")
    print(f"  - CSV Conflicts:          {os.path.join(output_dir, 'portfolio_report_conflicts.csv')}")
    print(f"  - Equity/Drawdown Chart:  {os.path.join(output_dir, 'equity_drawdown.png')}")
    print(f"  - Monthly Heatmap Chart:  {os.path.join(output_dir, 'monthly_heatmap.png')}")
    print(f"  - Yearly Returns Chart:   {os.path.join(output_dir, 'yearly_returns.png')}")
    print(f"  - Capital Usage Chart:    {os.path.join(output_dir, 'capital_usage.png')}")
    if args.optimize:
        print(f"  - Efficient Frontier:     {os.path.join(output_dir, 'efficient_frontier.png')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
