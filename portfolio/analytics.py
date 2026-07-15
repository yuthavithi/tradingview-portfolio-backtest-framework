"""
Portfolio Analytics Module.

This module provides functions and classes to analyze backtest performance,
compute risk-adjusted metrics, and generate beautiful visualizations.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from portfolio.importer import Trade

# Setup logger for the module
logger = logging.getLogger("portfolio.analytics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PortfolioAnalytics:
    """
    Computes performance metrics and generates visualizations for portfolio backtests.
    """

    def __init__(
        self,
        equity_curve: Optional[List[Tuple[datetime, float]]] = None,
        trades: Optional[List[Trade]] = None,
        initial_equity: Optional[float] = None,
        trading_days: int = 252,
    ) -> None:
        """
        Initializes the Portfolio Analytics engine.

        Args:
            equity_curve: List of (timestamp, equity) tuples.
            trades: List of parsed Trade dataclass objects.
            initial_equity: Override for initial capital (defaults to first equity value).
            trading_days: Periodic annualization factor (e.g. 252 for daily stock, 365 for crypto).
        """
        self.raw_equity_curve = equity_curve or []
        self.trades = trades or []
        self.trading_days = trading_days

        # Resampled Daily Equity and Returns Series
        self.equity_series = pd.Series(dtype=float)
        self.daily_returns = pd.Series(dtype=float)
        self.initial_equity = initial_equity

        self._prepare_equity_data()

    def _prepare_equity_data(self) -> None:
        """Processes and resamples raw equity curve into regular daily intervals."""
        if not self.raw_equity_curve:
            logger.warning("No equity curve data provided to PortfolioAnalytics.")
            return

        # 1. Convert to DataFrame
        df = pd.DataFrame(self.raw_equity_curve, columns=["timestamp", "equity"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df = df.set_index("timestamp")

        # 2. Resample to daily frequency using forward fill
        self.equity_series = df["equity"].resample("D").ffill()

        if self.initial_equity is None and not self.equity_series.empty:
            self.initial_equity = float(self.equity_series.iloc[0])

        # 3. Calculate daily percentage returns
        if len(self.equity_series) > 1:
            self.daily_returns = self.equity_series.pct_change().dropna()
        else:
            self.daily_returns = pd.Series(dtype=float)

    # ==========================================================================
    # Metric Calculations
    # ==========================================================================

    def calculate_cagr(self) -> float:
        """Calculates Compound Annual Growth Rate (CAGR)."""
        if self.equity_series.empty or len(self.equity_series) < 2:
            return 0.0
        
        start_val = self.initial_equity if self.initial_equity is not None else self.equity_series.iloc[0]
        end_val = self.equity_series.iloc[-1]
        
        if start_val <= 0 or end_val <= 0:
            return 0.0

        start_date = self.equity_series.index[0]
        end_date = self.equity_series.index[-1]
        duration_days = (end_date - start_date).days
        years = duration_days / 365.25

        if years <= 0:
            return 0.0

        try:
            return (end_val / start_val) ** (1.0 / years) - 1.0
        except (OverflowError, ZeroDivisionError):
            return 0.0

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculates the annualized Sharpe Ratio."""
        if self.daily_returns.empty or self.daily_returns.std() == 0:
            return 0.0
        
        daily_rf = risk_free_rate / self.trading_days
        excess_returns = self.daily_returns - daily_rf
        mean_excess = excess_returns.mean()
        std_returns = self.daily_returns.std()

        return (mean_excess / std_returns) * np.sqrt(self.trading_days)

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        """Calculates the annualized Sortino Ratio."""
        if self.daily_returns.empty:
            return 0.0

        daily_rf = risk_free_rate / self.trading_days
        excess_returns = self.daily_returns - daily_rf
        
        # Calculate downside deviation relative to target return
        downside_diff = self.daily_returns - target_return
        downside_diff_sq = np.where(downside_diff < 0, downside_diff ** 2, 0.0)
        downside_deviation = np.sqrt(np.mean(downside_diff_sq))

        if downside_deviation == 0:
            return 0.0

        return (excess_returns.mean() / downside_deviation) * np.sqrt(self.trading_days)

    def calculate_max_drawdown_percent(self) -> float:
        """Calculates the maximum drawdown percentage as a positive fraction."""
        if self.equity_series.empty:
            return 0.0
        peaks = self.equity_series.cummax()
        drawdowns = (peaks - self.equity_series) / peaks
        return float(drawdowns.max())

    def calculate_max_drawdown_cash(self) -> float:
        """Calculates the maximum drawdown in currency units."""
        if self.equity_series.empty:
            return 0.0
        peaks = self.equity_series.cummax()
        drawdowns = peaks - self.equity_series
        return float(drawdowns.max())

    def calculate_calmar_ratio(self) -> float:
        """Calculates the Calmar Ratio (CAGR / Max Drawdown)."""
        max_dd = self.calculate_max_drawdown_percent()
        if max_dd == 0:
            return float("inf")
        return self.calculate_cagr() / max_dd

    def calculate_mar_ratio(self) -> float:
        """Calculates the MAR Ratio (equal to Calmar based on CAGR)."""
        return self.calculate_calmar_ratio()

    def calculate_recovery_factor(self) -> float:
        """Calculates the Recovery Factor (Net Profit / Max Drawdown in cash)."""
        if self.equity_series.empty:
            return 0.0
        net_profit = self.equity_series.iloc[-1] - self.initial_equity
        max_dd_cash = self.calculate_max_drawdown_cash()
        if max_dd_cash == 0:
            return float("inf") if net_profit > 0 else 0.0
        return net_profit / max_dd_cash

    def calculate_ulcer_index(self) -> float:
        """Calculates the Ulcer Index (UI) measuring drawdown depth and duration."""
        if self.equity_series.empty:
            return 0.0
        peaks = self.equity_series.cummax()
        dd_pct = (peaks - self.equity_series) / peaks * 100.0
        return float(np.sqrt(np.mean(dd_pct ** 2)))

    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculates performance statistics from the list of trades."""
        # Chronological sort by exit time
        sorted_trades = sorted(self.trades, key=lambda t: t.exit_time or t.entry_time)

        total_trades = len(sorted_trades)
        if total_trades == 0:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
            }

        profits = [t.profit for t in sorted_trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        win_count = len(wins)
        win_rate = (win_count / total_trades) * 100.0

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0  # Kept negative
        largest_win = float(np.max(wins)) if wins else 0.0
        largest_loss = float(np.min(losses)) if losses else 0.0

        total_gain = sum(wins)
        total_pain = sum(losses)
        profit_factor = total_gain / abs(total_pain) if total_pain != 0 else float("inf")

        # Expectancy = (WinRate * AvgWin) + (LossRate * AvgLoss)
        loss_rate = (total_trades - win_count) / total_trades
        expectancy = (win_rate / 100.0 * avg_win) + (loss_rate * avg_loss)

        # Streaks
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for p in profits:
            if p > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif p < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
        }

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculates and consolidates all risk and trade metrics."""
        trade_stats = self.calculate_trade_metrics()
        cagr = self.calculate_cagr()
        max_dd_pct = self.calculate_max_drawdown_percent()

        metrics = {
            "initial_equity": self.initial_equity,
            "ending_equity": self.equity_series.iloc[-1] if not self.equity_series.empty else self.initial_equity,
            "net_profit": (self.equity_series.iloc[-1] - self.initial_equity) if not self.equity_series.empty else 0.0,
            "cagr": cagr,
            "sharpe": self.calculate_sharpe_ratio(),
            "sortino": self.calculate_sortino_ratio(),
            "max_drawdown_pct": max_dd_pct * 100.0,
            "max_drawdown_cash": self.calculate_max_drawdown_cash(),
            "calmar": self.calculate_calmar_ratio(),
            "mar": self.calculate_mar_ratio(),
            "recovery_factor": self.calculate_recovery_factor(),
            "ulcer_index": self.calculate_ulcer_index(),
        }
        
        # Merge trade stats
        metrics.update(trade_stats)
        return metrics

    # ==========================================================================
    # Visualizations (Plots)
    # ==========================================================================

    def plot_equity_curve(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Generates an Equity Curve plot."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        if not self.equity_series.empty:
            ax.plot(self.equity_series.index, self.equity_series, color="#1f77b4", linewidth=2, label="Equity")
            ax.fill_between(self.equity_series.index, self.equity_series, self.initial_equity, color="#1f77b4", alpha=0.1)
            ax.set_title("Equity Curve", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Date")
            ax.set_ylabel("Account Value ($)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left")
            plt.xticks(rotation=15)
            
        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def plot_drawdown_curve(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Generates a Drawdown Curve plot."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            
        if not self.equity_series.empty:
            peaks = self.equity_series.cummax()
            dd_pct = (self.equity_series - peaks) / peaks * 100.0
            
            ax.plot(dd_pct.index, dd_pct, color="#d62728", linewidth=1.5, label="Drawdown %")
            ax.fill_between(dd_pct.index, dd_pct, 0, color="#d62728", alpha=0.2)
            ax.set_title("Drawdown Curve", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Date")
            ax.set_ylabel("Drawdown (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower left")
            plt.xticks(rotation=15)
            
        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def _get_monthly_returns_matrix(self) -> pd.DataFrame:
        """Constructs a Year x Month returns matrix."""
        if self.equity_series.empty:
            return pd.DataFrame()
            
        # Get end of month equities
        monthly_equity = self.equity_series.resample("ME").last()
        
        # Calculate monthly percentage returns
        monthly_returns = monthly_equity.pct_change()
        
        # Resolve the first month's return relative to initial equity
        if not monthly_returns.empty:
            first_idx = monthly_returns.index[0]
            monthly_returns.loc[first_idx] = (monthly_equity.iloc[0] - self.initial_equity) / self.initial_equity

        # Group by Year and Month
        df_ret = pd.DataFrame({
            "Year": monthly_returns.index.year,
            "Month": monthly_returns.index.month,
            "Return": monthly_returns.values * 100.0
        })

        # Pivot to Year x Month
        pivot_df = df_ret.pivot(index="Year", columns="Month", values="Return")
        
        # Ensure all months 1-12 are represented columns
        for m in range(1, 13):
            if m not in pivot_df.columns:
                pivot_df[m] = np.nan
        pivot_df = pivot_df.reindex(columns=range(1, 13))
        
        # Rename columns to standard abbreviations
        pivot_df.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return pivot_df

    def plot_monthly_return_heatmap(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Generates a Heatmap of monthly performance returns."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        pivot_df = self._get_monthly_returns_matrix()
        if not pivot_df.empty:
            # Drop rows with all NaN
            pivot_clean = pivot_df.dropna(how="all")
            
            # Format matrix with custom diverging colors (red-yellow-green)
            sns.heatmap(
                pivot_clean,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                linewidths=0.5,
                ax=ax,
                cbar_kws={"label": "Return (%)"}
            )
            ax.grid(False)
            ax.set_title("Monthly Return Heat Map (%)", fontsize=14, fontweight="bold", pad=10)
            ax.set_ylabel("Year")
            ax.set_xlabel("Month")
            
        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def plot_yearly_returns(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Generates a Bar Chart of yearly returns."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            
        if not self.equity_series.empty:
            yearly_equity = self.equity_series.resample("YE").last()
            yearly_returns = yearly_equity.pct_change()
            
            # Calculate the first year's return
            if not yearly_returns.empty:
                first_year_idx = yearly_returns.index[0]
                yearly_returns.loc[first_year_idx] = (yearly_equity.iloc[0] - self.initial_equity) / self.initial_equity
            
            yearly_returns_pct = yearly_returns * 100.0
            years = yearly_returns_pct.index.year.astype(str)
            colors = ["#2ca02c" if val >= 0 else "#d62728" for val in yearly_returns_pct]
            
            bars = ax.bar(years, yearly_returns_pct, color=colors, edgecolor="black", width=0.5)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title("Yearly Return (%)", fontsize=14, fontweight="bold", pad=10)
            ax.set_ylabel("Return (%)")
            ax.set_xlabel("Year")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -13),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold"
                )
                
        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def plot_rolling_returns(self, window_days: int = 30, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Generates a rolling return plot over a window period (default 30 days)."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            
        if len(self.equity_series) > window_days:
            # Rolling return calculation: (Equity_t - Equity_t-w) / Equity_t-w
            rolling_ret = self.equity_series.pct_change(periods=window_days) * 100.0
            
            ax.plot(rolling_ret.index, rolling_ret, color="#9467bd", linewidth=1.5, label=f"{window_days}-Day Rolling Return")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"Rolling Return ({window_days}-Day Window)", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Date")
            ax.set_ylabel("Return (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper left")
            plt.xticks(rotation=15)
            
        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def plot_rolling_drawdown(self, window_days: int = 30, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Generates a rolling maximum drawdown plot over a window period."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            
        if len(self.equity_series) > window_days:
            # Rolling maximum drawdown inside window
            def rolling_dd(window_series):
                window_peaks = window_series.cummax()
                window_dds = (window_peaks - window_series) / window_peaks * 100.0
                return window_dds.max()
                
            rolling_max_dd = self.equity_series.rolling(window_days).apply(rolling_dd)
            
            # Drawdown plotted as negative values for visual consistency
            ax.plot(rolling_max_dd.index, -rolling_max_dd, color="#ff7f0e", linewidth=1.5, label=f"Rolling Max DD")
            ax.fill_between(rolling_max_dd.index, -rolling_max_dd, 0, color="#ff7f0e", alpha=0.15)
            ax.set_title(f"Rolling Drawdown ({window_days}-Day Window)", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Date")
            ax.set_ylabel("Max Drawdown (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower left")
            plt.xticks(rotation=15)
            
        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def plot_return_distribution(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Generates a Return Distribution curve (KDE + Density)."""
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            
        if not self.daily_returns.empty:
            returns_pct = self.daily_returns * 100.0
            sns.kdeplot(returns_pct, fill=True, color="#17becf", ax=ax, alpha=0.4, label="KDE Density")
            ax.axvline(returns_pct.mean(), color="#d62728", linestyle="--", label=f"Mean ({returns_pct.mean():.3f}%)")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Daily Return Distribution (KDE)", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel("Daily Return (%)")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            
        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def plot_histogram(self, data_type: str = "daily", ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Generates a histogram plot.

        Args:
            data_type: "daily" for daily returns, or "trade" for trade-by-trade profits.
            ax: Optional matplotlib axis.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        if data_type == "daily" and not self.daily_returns.empty:
            data = self.daily_returns * 100.0
            bins = 50
            label = "Daily Returns (%)"
            title = "Daily Returns Histogram"
        elif data_type == "trade" and self.trades:
            data = pd.Series([t.profit for t in self.trades])
            bins = min(20, len(data))
            label = "Trade Profit/Loss ($)"
            title = "Trade Performance Histogram"
        else:
            data = pd.Series(dtype=float)
            label = ""
            title = "Histogram (No Data)"

        if not data.empty:
            ax.hist(data, bins=bins, edgecolor="black", color="#34495e", alpha=0.8, rwidth=0.95)
            ax.axvline(data.mean(), color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Avg: {data.mean():.2f}")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()

        if fig is not None:
            plt.tight_layout()
            return fig
        return plt.gcf()

    def generate_plots(self, output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generates all individual analysis plots and saves them as files if output_dir is provided.

        Returns:
            Dict mapping plot name to its Matplotlib Figure object.
        """
        plots = {}
        
        # Temporary disable interactive display to prevent showing figures in loop
        plt.ioff()

        # Equity Curve
        fig_eq = plt.figure(figsize=(10, 5))
        self.plot_equity_curve(fig_eq.gca())
        plots["equity_curve"] = fig_eq

        # Drawdown Curve
        fig_dd = plt.figure(figsize=(10, 4))
        self.plot_drawdown_curve(fig_dd.gca())
        plots["drawdown_curve"] = fig_dd

        # Heatmap
        fig_hm = plt.figure(figsize=(10, 5))
        self.plot_monthly_return_heatmap(fig_hm.gca())
        plots["monthly_heatmap"] = fig_hm

        # Yearly
        fig_yr = plt.figure(figsize=(8, 4))
        self.plot_yearly_returns(fig_yr.gca())
        plots["yearly_returns"] = fig_yr

        # Rolling Return
        fig_rr = plt.figure(figsize=(10, 4))
        self.plot_rolling_returns(window_days=30, ax=fig_rr.gca())
        plots["rolling_returns"] = fig_rr

        # Rolling Drawdown
        fig_rdd = plt.figure(figsize=(10, 4))
        self.plot_rolling_drawdown(window_days=30, ax=fig_rdd.gca())
        plots["rolling_drawdown"] = fig_rdd

        # Distribution
        fig_dist = plt.figure(figsize=(8, 4))
        self.plot_return_distribution(fig_dist.gca())
        plots["return_distribution"] = fig_dist

        # Daily Histogram
        fig_hist = plt.figure(figsize=(8, 4))
        self.plot_histogram(data_type="daily", ax=fig_hist.gca())
        plots["daily_histogram"] = fig_hist

        # Trade Histogram
        if self.trades:
            fig_thist = plt.figure(figsize=(8, 4))
            self.plot_histogram(data_type="trade", ax=fig_thist.gca())
            plots["trade_histogram"] = fig_thist

        # Save to disk
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for name, fig in plots.items():
                fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
                logger.info(f"Saved analytics plot: {output_dir}/{name}.png")

        plt.ion()
        return plots

    def generate_dashboard(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        Generates a comprehensive performance analytics dashboard combining plots and stats.
        """
        plt.ioff()
        
        # Create a beautiful 3x3 dashboard grid layout
        fig = plt.figure(figsize=(18, 14), facecolor="#f8f9fa")
        grid = plt.GridSpec(3, 3, wspace=0.3, hspace=0.45)

        # Plot 1: Equity Curve
        ax_eq = fig.add_subplot(grid[0, :2])
        self.plot_equity_curve(ax_eq)

        # Performance Summary Panel (Text Box in grid cell 0, 2)
        ax_summary = fig.add_subplot(grid[0, 2])
        ax_summary.axis("off")
        ax_summary.set_facecolor("#ffffff")
        
        metrics = self.calculate_all_metrics()
        
        summary_text = (
            f"  PORTFOLIO PERFORMANCE SUMMARY\n"
            f"  ==============================\n\n"
            f"  Initial Capital:   ${metrics['initial_equity']:,.2f}\n"
            f"  Ending Capital:    ${metrics['ending_equity']:,.2f}\n"
            f"  Net Profit:        ${metrics['net_profit']:+,.2f}\n"
            f"  CAGR:              {metrics['cagr']*100:.2f}%\n"
            f"  Max Drawdown:      {metrics['max_drawdown_pct']:.2f}%\n"
            f"  Sharpe Ratio:      {metrics['sharpe']:.2f}\n"
            f"  Sortino Ratio:     {metrics['sortino']:.2f}\n"
            f"  Calmar Ratio:      {metrics['calmar']:.2f}\n"
            f"  Ulcer Index:       {metrics['ulcer_index']:.2f}\n"
            f"  Recovery Factor:   {metrics['recovery_factor']:.2f}\n\n"
            f"  TRADE STATISTICS\n"
            f"  ==============================\n\n"
            f"  Total Trades:      {len(self.trades)}\n"
            f"  Win Rate:          {metrics['win_rate']:.1f}%\n"
            f"  Profit Factor:     {metrics['profit_factor']:.2f}\n"
            f"  Expectancy:        ${metrics['expectancy']:+,.2f}\n"
            f"  Avg Win:           ${metrics['avg_win']:,.2f}\n"
            f"  Avg Loss:          ${metrics['avg_loss']:,.2f}\n"
            f"  Max Win:           ${metrics['largest_win']:,.2f}\n"
            f"  Max Loss:          ${metrics['largest_loss']:,.2f}\n"
            f"  Max Streak Wins:   {metrics['max_consecutive_wins']}\n"
            f"  Max Streak Losses: {metrics['max_consecutive_losses']}\n"
        )
        
        ax_summary.text(
            0.05, 0.95, summary_text,
            transform=ax_summary.transAxes,
            fontsize=10.5,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="white", edgecolor="#cccccc", alpha=0.9)
        )

        # Plot 2: Drawdown Curve
        ax_dd = fig.add_subplot(grid[1, 0])
        self.plot_drawdown_curve(ax_dd)

        # Plot 3: Return Distribution
        ax_dist = fig.add_subplot(grid[1, 1])
        self.plot_return_distribution(ax_dist)

        # Plot 4: Yearly Returns
        ax_yr = fig.add_subplot(grid[1, 2])
        self.plot_yearly_returns(ax_yr)

        # Plot 5: Monthly Heatmap
        ax_hm = fig.add_subplot(grid[2, 0])
        self.plot_monthly_return_heatmap(ax_hm)

        # Plot 6: Rolling Drawdown
        ax_rdd = fig.add_subplot(grid[2, 1])
        self.plot_rolling_drawdown(window_days=30, ax=ax_rdd)

        # Plot 7: Rolling Return
        ax_rr = fig.add_subplot(grid[2, 2])
        self.plot_rolling_returns(window_days=30, ax=ax_rr)

        fig.suptitle("Portfolio Analytics Dashboard", fontsize=20, fontweight="bold", y=0.96)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved performance dashboard: {output_path}")

        plt.ion()
        return fig


# ==============================================================================
# Runnable Demo / Example
# ==============================================================================

def run_example() -> None:
    """Generates mock performance data and builds the dashboard."""
    print("--- Running Portfolio Analytics Visualization Example ---")
    
    # 1. Create a synthetic daily equity curve over 2 years (approx 730 days)
    base_time = datetime(2024, 1, 1)
    np.random.seed(42)
    
    # Simulate random walk with drift (e.g. profitable trading strategy)
    n_days = 730
    daily_pct = np.random.normal(loc=0.0006, scale=0.012, size=n_days)
    
    # Inject a few larger drawdown events to make it realistic
    daily_pct[150:180] -= 0.008  # 30-day drawdown
    daily_pct[450:480] -= 0.007  # another drawdown
    
    equity = [10000.0]
    for p in daily_pct:
        next_eq = equity[-1] * (1.0 + p)
        equity.append(next_eq)
        
    equity_curve = [
        (base_time + pd.Timedelta(days=i), eq) for i, eq in enumerate(equity)
    ]
    
    # 2. Create synthetic trades
    trades = []
    # 100 random trades with 55% win rate
    for idx in range(1, 101):
        is_win = np.random.rand() < 0.55
        val = np.random.exponential(scale=150.0)
        profit = val if is_win else -val * 0.8  # positive expectancy
        
        trades.append(
            Trade(
                strategy_name="AlphaSystem",
                trade_id=idx,
                entry_time=base_time + pd.Timedelta(days=idx * 7),
                exit_time=base_time + pd.Timedelta(days=idx * 7 + 2),
                side="Long",
                entry_price=100.0,
                exit_price=100.0 + (profit / 10.0),
                contracts=10.0,
                position_value=1000.0,
                commission=1.5,
                profit=profit - 3.0,  # includes commissions
                profit_percent=(profit / 1000.0) * 100.0,
                holding_time=pd.Timedelta(days=2)
            )
        )
        
    # 3. Instantiate analytics engine
    analytics = PortfolioAnalytics(equity_curve=equity_curve, trades=trades, initial_equity=10000.0)
    
    # 4. Generate all metrics
    metrics = analytics.calculate_all_metrics()
    print("\nCalculated Portfolio Metrics:")
    print(f"  Initial Equity:    ${metrics['initial_equity']:.2f}")
    print(f"  Ending Equity:     ${metrics['ending_equity']:.2f}")
    print(f"  Net Profit:        ${metrics['net_profit']:.2f}")
    print(f"  CAGR:              {metrics['cagr'] * 100:.2f}%")
    print(f"  Max Drawdown %:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe']:.2f}")
    print(f"  Sortino Ratio:     {metrics['sortino']:.2f}")
    print(f"  Calmar Ratio:      {metrics['calmar']:.2f}")
    print(f"  Ulcer Index:       {metrics['ulcer_index']:.2f}")
    print(f"  Recovery Factor:   {metrics['recovery_factor']:.2f}")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"  Expectancy:        ${metrics['expectancy']:.2f}")
    print(f"  Win Rate:          {metrics['win_rate']:.1f}%")
    print(f"  Max Consecutive Wins: {metrics['max_consecutive_wins']}")
    print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")
    
    # 5. Generate and Save plots/dashboard
    os.makedirs("output_plots", exist_ok=True)
    
    print("\nGenerating dashboard...")
    analytics.generate_dashboard(output_path="output_plots/dashboard.png")
    
    print("Generating individual plots...")
    analytics.generate_plots(output_dir="output_plots")
    
    print("\nDone! Visualizations generated successfully in directory 'output_plots'.")


if __name__ == "__main__":
    run_example()
