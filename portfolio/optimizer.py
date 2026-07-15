"""
Portfolio Optimizer Module.

This module implements analytical and simulation-based portfolio optimization
for TradingView strategies sharing a single account.
It uses scipy.optimize to find optimal strategy weights for various objectives
subject to leverage, weight, risk, and cash constraints, and verifies results
using the event-driven SharedCapitalEngine.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.optimize as sco

from portfolio.importer import Trade
from portfolio.shared_engine import SharedCapitalEngine
from portfolio.analytics import PortfolioAnalytics
from portfolio.risk import RiskParameters, RiskEngine

# Setup logger for the module
logger = logging.getLogger("portfolio.optimizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PortfolioOptimizer:
    """
    Optimizes portfolio weights across multiple TradingView strategies.
    Supports analytical returns-based optimization and event-driven verification.
    """

    def __init__(
        self,
        strategies: Dict[str, List[Trade]],
        initial_equity: float = 100000.0,
        leverage: float = 1.0,
        risk_free_rate: float = 0.0,
        trading_days: int = 252,
        stress_test_drawdown: bool = False,
    ) -> None:
        """
        Initializes the PortfolioOptimizer.

        Args:
            strategies: Dict mapping strategy name to list of Trade objects.
            initial_equity: Starting capital for the portfolio.
            leverage: Leverage factor used for individual strategy simulation.
            risk_free_rate: Annualized risk-free rate.
            trading_days: Annualization factor (e.g. 252 for daily stock/futures, 365 for crypto).
        """
        self.strategies = strategies
        self.initial_equity = initial_equity
        self.leverage = leverage
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.stress_test_drawdown = stress_test_drawdown

        self.strategy_names = list(strategies.keys())
        if not self.strategy_names:
            raise ValueError("At least one strategy must be provided.")

        # Matrices for fast analytical calculations
        self.equity_matrix: np.ndarray = np.array([])
        self.cash_matrix: np.ndarray = np.array([])
        self.margin_matrix: np.ndarray = np.array([])
        self.avail_margin_matrix: np.ndarray = np.array([])
        self.positions_matrix: np.ndarray = np.array([])
        self.returns_matrix: np.ndarray = np.array([])
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex([])

        # Reconstruct base daily curves
        self._reconstruct_base_curves()

    def _reconstruct_base_curves(self) -> None:
        """Runs each strategy independently to reconstruct its daily equity and cash metrics."""
        logger.info("Reconstructing base equity curves for all strategies...")
        dfs = []

        for name in self.strategy_names:
            trades = self.strategies[name]
            # Run engine with full capital for the single strategy
            engine = SharedCapitalEngine(
                initial_equity=self.initial_equity,
                leverage=self.leverage,
                stress_test_drawdown=self.stress_test_drawdown,
            )
            res = engine.run(trades)
            history = engine.history

            if not history:
                logger.warning(f"No simulation history generated for strategy '{name}'.")
                continue

            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            df = df.set_index("timestamp")

            # Resample to daily frequency using forward fill
            df_daily = df.resample("D").ffill()
            df_daily = df_daily.rename(columns={
                "equity": f"equity_{name}",
                "cash": f"cash_{name}",
                "margin_used": f"margin_{name}",
                "available_margin": f"avail_margin_{name}",
                "concurrent_positions": f"positions_{name}"
            })
            dfs.append(df_daily)

        if not dfs:
            raise ValueError("Could not reconstruct simulation history for any strategy.")

        # Reindex all to a unified date range
        all_dates = pd.DatetimeIndex([])
        for df in dfs:
            all_dates = all_dates.union(df.index)

        self.dates = all_dates
        aligned_dfs = []

        for name, df in zip(self.strategy_names, dfs):
            df_reindexed = df.reindex(all_dates)
            
            # Fill pre-trading periods with defaults
            first_idx = df_reindexed[f"equity_{name}"].first_valid_index()
            if first_idx is not None:
                df_reindexed.loc[:first_idx, f"equity_{name}"] = df_reindexed.loc[:first_idx, f"equity_{name}"].fillna(self.initial_equity)
                df_reindexed.loc[:first_idx, f"cash_{name}"] = df_reindexed.loc[:first_idx, f"cash_{name}"].fillna(self.initial_equity)
                df_reindexed.loc[:first_idx, f"avail_margin_{name}"] = df_reindexed.loc[:first_idx, f"avail_margin_{name}"].fillna(self.initial_equity)
                df_reindexed.loc[:first_idx, f"margin_{name}"] = df_reindexed.loc[:first_idx, f"margin_{name}"].fillna(0.0)
                df_reindexed.loc[:first_idx, f"positions_{name}"] = df_reindexed.loc[:first_idx, f"positions_{name}"].fillna(0)

            # Forward fill the rest
            df_reindexed = df_reindexed.ffill()
            aligned_dfs.append(df_reindexed)

        df_merged = pd.concat(aligned_dfs, axis=1)

        # Populate matrices
        self.equity_matrix = df_merged[[f"equity_{name}" for name in self.strategy_names]].values
        self.cash_matrix = df_merged[[f"cash_{name}" for name in self.strategy_names]].values
        self.margin_matrix = df_merged[[f"margin_{name}" for name in self.strategy_names]].values
        self.avail_margin_matrix = df_merged[[f"avail_margin_{name}" for name in self.strategy_names]].values
        self.positions_matrix = df_merged[[f"positions_{name}" for name in self.strategy_names]].values

        # Compute returns matrix
        returns_list = []
        for name in self.strategy_names:
            eq_series = df_merged[f"equity_{name}"]
            # Daily returns
            ret_series = eq_series.pct_change().fillna(0.0)
            returns_list.append(ret_series)
        df_returns = pd.concat(returns_list, axis=1)
        self.returns_matrix = df_returns.values

        logger.info(f"Reconstructed and aligned curves over {len(self.dates)} days.")

    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Calculates analytical CAGR, Max Drawdown, Sharpe, Calmar, and Recovery Factor for a weight vector.

        Args:
            weights: Numpy array of strategy weights.

        Returns:
            A tuple of (cagr, max_drawdown, sharpe, calmar, recovery_factor).
        """
        C0 = self.initial_equity
        # E_p(t) = C0 + sum_i w_i * (E_i(t) - C0)
        equity_diff = self.equity_matrix - C0
        E_p = C0 + equity_diff.dot(weights)

        # Check for bankruptcy
        if np.any(E_p <= 0):
            return -1.0, 1.0, -10.0, 0.0, 0.0

        # Calculate returns
        R_p = np.diff(E_p) / E_p[:-1] if len(E_p) > 1 else np.array([0.0])

        # CAGR
        duration_days = (self.dates[-1] - self.dates[0]).days
        years = duration_days / 365.25
        if years > 0 and E_p[-1] > 0:
            cagr = (E_p[-1] / C0) ** (1.0 / years) - 1.0
        else:
            cagr = 0.0

        # Max Drawdown (percentage)
        peaks = np.maximum.accumulate(E_p)
        drawdowns = (peaks - E_p) / peaks
        max_dd = np.max(drawdowns)

        # Sharpe Ratio
        mean_ret = np.mean(R_p) - (self.risk_free_rate / self.trading_days)
        std_ret = np.std(R_p)
        sharpe = (mean_ret / std_ret) * np.sqrt(self.trading_days) if std_ret > 0 else 0.0

        # Calmar / MAR Ratio
        calmar = cagr / max_dd if max_dd > 0 else 0.0

        # Recovery Factor (Net Profit / Max Cash Drawdown)
        net_profit = E_p[-1] - C0
        cash_drawdowns = peaks - E_p
        max_dd_cash = np.max(cash_drawdowns)
        recovery_factor = net_profit / max_dd_cash if max_dd_cash > 0 else 0.0

        return cagr, max_dd, sharpe, calmar, recovery_factor

    def _calculate_portfolio_sortino(self, weights: np.ndarray) -> float:
        """
        Calculates the annualized Sortino Ratio for a given set of weights.
        """
        C0 = self.initial_equity
        equity_diff = self.equity_matrix - C0
        E_p = C0 + equity_diff.dot(weights)

        # Check for bankruptcy
        if np.any(E_p <= 0):
            return -10.0

        # Calculate returns
        R_p = np.diff(E_p) / E_p[:-1] if len(E_p) > 1 else np.array([0.0])

        mean_ret = np.mean(R_p) - (self.risk_free_rate / self.trading_days)
        
        # Calculate downside deviation relative to target return (0.0)
        downside_diff = R_p - 0.0
        downside_diff_sq = np.where(downside_diff < 0, downside_diff ** 2, 0.0)
        downside_deviation = np.sqrt(np.mean(downside_diff_sq))

        sortino = (mean_ret / downside_deviation) * np.sqrt(self.trading_days) if downside_deviation > 0 else 0.0
        return sortino

    def _calculate_monte_carlo_cvar(self, w: np.ndarray, iterations: int = 1000, confidence: float = 0.95) -> float:
        """
        Calculates the Conditional Drawdown at Risk (CVaR of Max Drawdown) using Bootstrap Resampling.
        
        Args:
            w: Array of strategy weights.
            iterations: Number of simulated Monte Carlo paths.
            confidence: Confidence level for CVaR (e.g., 0.95 for worst 5%).
            
        Returns:
            The expected CVaR (as a positive float).
        """
        if len(self.returns_matrix) == 0:
            return 0.0

        # Calculate base portfolio daily returns
        R_p = self.returns_matrix.dot(w)
        N_days = len(R_p)
        
        if N_days == 0:
            return 0.0

        # Vectorized Monte Carlo Resampling
        # Generate random indices of shape (iterations, N_days) with replacement
        random_indices = np.random.randint(0, N_days, size=(iterations, N_days))
        
        # Resampled return paths of shape (iterations, N_days)
        simulated_returns = R_p[random_indices]
        
        # Compute cumulative equity paths
        equity_paths = self.initial_equity * np.cumprod(1 + simulated_returns, axis=1)
        
        # running max of equity path
        running_max = np.maximum.accumulate(equity_paths, axis=1)
        
        # Calculate drawdowns and max drawdown for each path
        drawdowns = (running_max - equity_paths) / running_max
        max_drawdowns = np.max(drawdowns, axis=1)
        
        # Sort maximum drawdowns in ascending order
        sorted_max_drawdowns = np.sort(max_drawdowns)
        
        # Determine the index for the tail
        tail_start_idx = int(iterations * confidence)
        
        # CVaR is the mean of the worst (1 - confidence) portion of maximum drawdowns
        tail_drawdowns = sorted_max_drawdowns[tail_start_idx:]
        if len(tail_drawdowns) == 0:
            cvar = sorted_max_drawdowns[-1]
        else:
            cvar = float(np.mean(tail_drawdowns))
            
        return cvar

    def optimize(
        self,
        objective: str = "sharpe",
        max_leverage: float = 1.0,
        max_portfolio_risk: Optional[float] = None,
        max_concurrent_positions: Optional[int] = None,
        min_cash_reserve: float = 0.0,
        weight_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        cvar_iterations: int = 1000,
        cvar_confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Solves for the optimal strategy weights analytically.

        Args:
            objective: Choice of 'cagr', 'drawdown', 'sharpe', 'calmar', 'mar', 'recovery_factor'.
            max_leverage: Maximum total leverage constraint (sum of weights).
            max_portfolio_risk: Maximum annualized portfolio volatility constraint.
            max_concurrent_positions: Maximum concurrent positions constraint.
            min_cash_reserve: Minimum cash/available margin reserve. If <= 1.0, treated as equity fraction.
            weight_bounds: Dict of strategy weights bounds. Defaults to (0.0, max_leverage).

        Returns:
            Dictionary containing optimized weights and expected performance metrics.
        """
        N = len(self.strategy_names)
        objective = objective.lower()

        # Parse minimum cash reserve absolute threshold
        cash_threshold = min_cash_reserve
        if min_cash_reserve > 0 and min_cash_reserve <= 1.0:
            cash_threshold = min_cash_reserve * self.initial_equity

        # Setup bounds
        bounds = []
        for name in self.strategy_names:
            if weight_bounds and name in weight_bounds:
                bounds.append(weight_bounds[name])
            else:
                bounds.append((0.0, max_leverage))

        # Initial guess: equal weights summing to 1.0 or min of bounds
        x0 = np.ones(N) / N

        # Objective function to minimize
        def loss_function(w: np.ndarray) -> float:
            cagr, max_dd, sharpe, calmar, rf = self._calculate_portfolio_metrics(w)
            
            # Penalize sum of weights exceeding leverage inside loss just in case
            pen = 0.0
            if np.sum(w) > max_leverage + 1e-5:
                pen += (np.sum(w) - max_leverage) * 100.0

            if objective == "cagr":
                return -cagr + pen
            elif objective == "drawdown" or objective == "min max drawdown":
                return max_dd + pen
            elif objective == "cvar":
                cvar = self._calculate_monte_carlo_cvar(w, iterations=cvar_iterations, confidence=cvar_confidence)
                return cvar + pen
            elif objective == "sharpe" or objective == "max sharpe":
                return -sharpe + pen
            elif objective == "sortino" or objective == "max sortino":
                sortino = self._calculate_portfolio_sortino(w)
                return -sortino + pen
            elif objective == "calmar" or objective == "mar" or objective == "max calmar" or objective == "max mar ratio":
                return -calmar + pen
            elif objective == "recovery_factor" or objective == "max recovery factor":
                return -rf + pen
            else:
                raise ValueError(f"Unsupported objective: {objective}")

        # Constraints list
        constraints = []

        # Leverage constraint: sum(w) <= max_leverage
        constraints.append({
            "type": "ineq",
            "fun": lambda w: max_leverage - np.sum(w)
        })

        # Minimum Cash Reserve / Available Margin constraint
        if cash_threshold > 0:
            def cash_constraint(w: np.ndarray) -> float:
                C0 = self.initial_equity
                avail_margin_diff = self.avail_margin_matrix - C0
                avail_margin_p = C0 + avail_margin_diff.dot(w)
                return np.min(avail_margin_p) - cash_threshold
            
            constraints.append({
                "type": "ineq",
                "fun": cash_constraint
            })

        # Maximum Portfolio Risk (annualized volatility)
        if max_portfolio_risk is not None and max_portfolio_risk > 0:
            def risk_constraint(w: np.ndarray) -> float:
                C0 = self.initial_equity
                equity_diff = self.equity_matrix - C0
                E_p = C0 + equity_diff.dot(w)
                if np.any(E_p <= 0):
                    return -1.0
                R_p = np.diff(E_p) / E_p[:-1] if len(E_p) > 1 else np.array([0.0])
                vol = np.std(R_p) * np.sqrt(self.trading_days)
                return max_portfolio_risk - vol

            constraints.append({
                "type": "ineq",
                "fun": risk_constraint
            })

        # Maximum Concurrent Positions
        if max_concurrent_positions is not None and max_concurrent_positions > 0:
            def positions_constraint(w: np.ndarray) -> float:
                # Approximate active strategies count based on weight threshold
                active_mask = (w > 1e-4).astype(float)
                max_pos = np.max(self.positions_matrix.dot(active_mask))
                return max_concurrent_positions - max_pos

            constraints.append({
                "type": "ineq",
                "fun": positions_constraint
            })

        # Perform minimization
        # Use SLSQP because it handles both bounds and constraints well
        res = sco.minimize(
            loss_function,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6}
        )

        if not res.success:
            logger.warning(f"Optimization did not converge cleanly: {res.message}. Using best solution.")

        optimal_weights = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
        # Clean small weights to zero
        optimal_weights[optimal_weights < 1e-4] = 0.0

        # Enforce exact leverage limit after cleaning
        if np.sum(optimal_weights) > max_leverage:
            optimal_weights = optimal_weights * (max_leverage / np.sum(optimal_weights))

        # Map to dict
        optimal_weights_dict = {
            self.strategy_names[i]: float(optimal_weights[i]) for i in range(N)
        }

        # Calculate final expected metrics
        cagr, max_dd, sharpe, calmar, rf = self._calculate_portfolio_metrics(optimal_weights)
        cvar = self._calculate_monte_carlo_cvar(optimal_weights, iterations=cvar_iterations, confidence=cvar_confidence)
        sortino = self._calculate_portfolio_sortino(optimal_weights)

        # Calculate expected equity curve
        C0 = self.initial_equity
        equity_diff = self.equity_matrix - C0
        expected_equity = C0 + equity_diff.dot(optimal_weights)
        expected_equity_curve = [(self.dates[i], float(expected_equity[i])) for i in range(len(self.dates))]

        # Generate report detail
        report = []
        for i, name in enumerate(self.strategy_names):
            w = optimal_weights[i]
            allocated_cap = w * self.initial_equity
            report.append({
                "strategy": name,
                "weight": float(w),
                "allocated_capital": float(allocated_cap)
            })

        return {
            "optimal_weights": optimal_weights_dict,
            "expected_cagr": float(cagr),
            "expected_max_drawdown": float(max_dd),
            "expected_cvar": float(cvar),
            "expected_sharpe": float(sharpe),
            "expected_sortino": float(sortino),
            "expected_calmar": float(calmar),
            "expected_recovery_factor": float(rf),
            "expected_equity_curve": expected_equity_curve,
            "capital_allocation_report": report,
            "success": bool(res.success),
            "message": str(res.message)
        }

    def generate_efficient_frontier(
        self,
        max_leverage: float = 1.0,
        min_cash_reserve: float = 0.0,
        weight_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        points_count: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Generates the Efficient Frontier (CAGR vs. Drawdown) by minimizing drawdown for target CAGR values.

        Returns:
            A list of dicts containing portfolio allocations, CAGRs, and Drawdowns along the frontier.
        """
        logger.info("Generating efficient frontier...")
        N = len(self.strategy_names)

        # 1. Determine min and max CAGR possible
        cagrs = []
        for name in self.strategy_names:
            w = np.zeros(N)
            w[self.strategy_names.index(name)] = max_leverage
            cagr, _, _, _, _ = self._calculate_portfolio_metrics(w)
            cagrs.append(cagr)

        min_cagr = max(0.0, min(cagrs))
        max_cagr = max(cagrs)

        if max_cagr <= min_cagr:
            # Fallback range if all strategies are similar
            min_cagr = 0.0
            max_cagr = 0.30

        target_cagrs = np.linspace(min_cagr, max_cagr, points_count)
        frontier_points = []

        # Setup bounds
        bounds = []
        for name in self.strategy_names:
            if weight_bounds and name in weight_bounds:
                bounds.append(weight_bounds[name])
            else:
                bounds.append((0.0, max_leverage))

        x0 = np.ones(N) / N

        for target in target_cagrs:
            # Minimize Max Drawdown
            def loss(w: np.ndarray) -> float:
                _, max_dd, _, _, _ = self._calculate_portfolio_metrics(w)
                return max_dd

            constraints = [
                # sum(w) <= max_leverage
                {"type": "ineq", "fun": lambda w: max_leverage - np.sum(w)},
                # CAGR >= target
                {"type": "ineq", "fun": lambda w: self._calculate_portfolio_metrics(w)[0] - target}
            ]

            # Cash Reserve constraint
            if min_cash_reserve > 0:
                cash_threshold = min_cash_reserve
                if min_cash_reserve <= 1.0:
                    cash_threshold = min_cash_reserve * self.initial_equity

                def cash_constraint(w: np.ndarray) -> float:
                    C0 = self.initial_equity
                    avail_margin_diff = self.avail_margin_matrix - C0
                    avail_margin_p = C0 + avail_margin_diff.dot(w)
                    return np.min(avail_margin_p) - cash_threshold

                constraints.append({"type": "ineq", "fun": cash_constraint})

            res = sco.minimize(
                loss,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-5}
            )

            if res.success:
                w_opt = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
                w_opt[w_opt < 1e-4] = 0.0
                if np.sum(w_opt) > max_leverage:
                    w_opt = w_opt * (max_leverage / np.sum(w_opt))

                actual_cagr, actual_dd, actual_sharpe, _, _ = self._calculate_portfolio_metrics(w_opt)
                frontier_points.append({
                    "target_cagr": float(target),
                    "cagr": float(actual_cagr),
                    "drawdown": float(actual_dd),
                    "sharpe": float(actual_sharpe),
                    "weights": {self.strategy_names[i]: float(w_opt[i]) for i in range(N)}
                })

        return frontier_points

    def verify_simulation(
        self,
        optimal_weights: Dict[str, float],
        risk_engine_params: Optional[RiskParameters] = None,
    ) -> Dict[str, Any]:
        """
        Validates the optimal weights by running them through the event-driven SharedCapitalEngine.
        Scales each strategy's trade properties (contracts, position_value, profit, commission) by its weight.

        Args:
            optimal_weights: Dict mapping strategy name to its weight.
            risk_engine_params: Optional RiskParameters to construct RiskEngine for simulation.

        Returns:
            The backtest results from SharedCapitalEngine.
        """
        logger.info("Running event-driven simulation to verify optimal weights...")
        
        # 1. Merge trades for active strategies
        merged_trades = []
        for name, trades in self.strategies.items():
            weight = optimal_weights.get(name, 0.0)
            if weight <= 0.0:
                continue
            merged_trades.extend(trades)

        # Sort trades chronologically by entry_time
        merged_trades = sorted(merged_trades, key=lambda t: t.entry_time)

        # 2. Run simulation
        engine = SharedCapitalEngine(
            initial_equity=self.initial_equity,
            leverage=self.leverage,
            strategy_weights=optimal_weights,
            stress_test_drawdown=self.stress_test_drawdown,
        )

        risk_engine = None
        if risk_engine_params is not None:
            # Ensure leverage matches
            risk_engine_params.leverage = self.leverage
            risk_engine = RiskEngine(risk_engine_params)

        sim_results = engine.run(merged_trades, risk_engine=risk_engine)
        sim_results["history"] = engine.history
        
        return sim_results

    def plot_efficient_frontier(
        self,
        frontier_points: List[Dict[str, Any]],
        optimal_point: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Generates and saves/shows a plot of the Efficient Frontier (CAGR vs. Drawdown)."""
        if not frontier_points:
            logger.warning("No efficient frontier points to plot.")
            return

        cagrs = [p["cagr"] * 100.0 for p in frontier_points]
        drawdowns = [p["drawdown"] * 100.0 for p in frontier_points]

        plt.figure(figsize=(10, 6))
        plt.plot(drawdowns, cagrs, "o-", color="#10b981", linewidth=2.5, markersize=6, label="Efficient Frontier")
        
        if optimal_point:
            opt_cagr = optimal_point.get("expected_cagr", 0.0) * 100.0
            opt_dd = optimal_point.get("expected_max_drawdown", 0.0) * 100.0
            plt.scatter([opt_dd], [opt_cagr], color="#ef4444", s=120, zorder=5, label="Optimal Allocation")
            plt.annotate(
                f"Optimal ({opt_dd:.1f}% DD, {opt_cagr:.1f}% CAGR)",
                xy=(opt_dd, opt_cagr),
                xytext=(opt_dd + 2, opt_cagr - 2),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=6),
                fontweight="bold"
            )

        plt.title("Efficient Frontier (CAGR vs. Max Drawdown)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Max Drawdown (%)", fontsize=12)
        plt.ylabel("Expected CAGR (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right", fontsize=11)
        plt.tight_layout()

        if save_path:
            # Ensure folder exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved Efficient Frontier plot to: {save_path}")
            plt.close()
        else:
            plt.show()
