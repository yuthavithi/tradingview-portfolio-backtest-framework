# TradingView Portfolio Backtest & Optimization Framework Master Prompt

Act as a lead quantitative software architect. Below is a comprehensive building specification to construct the **TradingView Portfolio Backtest & Optimization Framework** in Python. This framework processes exported TradingView backtest Excel files, simulates strategy execution under a shared account with leverage-based margin constraints, resolves capital allocation conflicts, manages risk, optimizes strategy weights using daily return bootstrap resampled Conditional Value at Risk (CVaR), and outputs professional PDF, Excel, and CSV performance reports.

---

## 1. Directory Structure

The project should be organized as follows:

```
├── main.py                     # CLI entrypoint and orchestrator
├── portfolio/
│   ├── __init__.py
│   ├── engine.py               # Generic event-driven broker simulator
│   ├── importer.py             # TradingView Excel importer
│   ├── shared_engine.py        # Event-driven shared capital simulation engine
│   ├── optimizer.py            # Portfolio weight optimizer & MC CVaR calculator
│   ├── analytics.py            # Advanced performance & risk ratios calculator
│   ├── risk.py                 # Risk Engine & limits manager
│   ├── report.py               # PDF, Excel, and CSV report compiler
│   ├── events.py               # Priority events definitions
│   └── queue.py                # Priority event queue
├── tests/                      # Unit and integration test suite
│   ├── test_importer.py
│   ├── test_portfolio_engine.py
│   ├── test_shared_engine.py
│   ├── test_risk.py
│   ├── test_optimizer.py
│   ├── test_analytics.py
│   ├── test_report.py
│   └── test_cli.py
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation
```

---

## 2. Dependencies (`requirements.txt`)

Ensure the following packages are specified:
```text
pandas>=2.2.2
numpy>=2.0.0
scipy>=1.14.0
matplotlib>=3.9.1
seaborn>=0.13.2
openpyxl>=3.1.5
reportlab>=4.1.0
pypdf>=4.0.0
pytest>=8.0.0
```

---

## 3. Detailed Component Specifications

### 3.1. Events & Priority Queue (`portfolio/events.py` & `portfolio/queue.py`)

*   **`events.py`**:
    *   Implement a `BaseEvent` dataclass containing `timestamp: Union[datetime, float, int]` and `event_type: str`.
    *   Override `__lt__` (less than) in `BaseEvent` so events are ordered chronologically by `timestamp`.
    *   **Exit Prioritization Rule**: If two events share the *exact same timestamp*, prioritize exits over entries to release margin first. This is achieved by comparing `event_type` alphabetically as a tie-breaker. Since `MarginReleaseEvent` (exits) is alphabetically smaller than `MarginRequestEvent` (entries), the exit event naturally pops first.
    *   Define event subclasses:
        *   `EntryEvent`, `ExitEvent`, `OrderEvent`, `TradeEvent`, `MarginEvent`, `AccountEvent`, `PortfolioEvent`, `MarketDataEvent` inheriting from `BaseEvent`.
    *   *(Note: `MarginRequestEvent` and `MarginReleaseEvent` are defined directly within `shared_engine.py`)*
        *   `MarginRequestEvent(BaseEvent)`: Tracks `trade_id`, `position_value`, `profit`, `side`, `strategy_name`, and a reference to the `Trade` object. Includes a dynamic `scale_factor` (default 1.0) to scale sizes when portfolio weights are modified.
        *   `MarginReleaseEvent(BaseEvent)`: Mirror of request event; used to release locked margin and realize profit/loss.
*   **`queue.py`**:
    *   Implement `EventQueue` wrapping Python's standard `heapq` module (Min-Heap). Provide `push`, `pop`, `peek`, `empty`, `clear`, and `__len__` methods.

### 3.2. TradingView Importer (`portfolio/importer.py`)

*   **Data Structures**:
    *   Define `Trade` dataclass with fields: `strategy_name: str`, `trade_id: Union[int, str]`, `entry_time: datetime`, `exit_time: datetime`, `side: str` ('Long' or 'Short'), `entry_price: float`, `exit_price: float`, `contracts: float`, `position_value: float` (entry_price * contracts), `commission: float`, `profit: float`, `profit_percent: float`, `holding_time: timedelta`, `mae: float` (Maximum Adverse Excursion), `initial_capital: Optional[float]`.
    *   Define `TradingViewReport` dataclass matching the structure of a single XLSX report containing file path, strategy name, settings dict, performance summary dataframe, trades list, and orders dataframe.
*   **Parsing Logic**:
    *   Support both English and Chinese TradingView Excel exports. Implement mapping helpers to locate sheet and column names based on candidate translations (e.g., matching sheets named `properties`, `settings`, `属性`, `设置` or columns named `net pnl`, `净损益`).
    *   Clean column names by converting them to lowercase, stripping whitespace, and removing parentheses/symbols (e.g., `($)` or `(%)`).
    *   TradingView lists trades across multiple rows (one entry fill row and one exit fill row). Implement logic to group rows by Trade ID (`trade number` / `trade #` / `编号`), identify entry vs exit timestamps, calculate weighted average entry/exit prices, sum contracts/commissions, extract MAE, and yield clean `Trade` objects.
    *   Support importing multiple files from a folder or glob pattern, handling strategy name generation cleanly (with deduplication suffix like `_1`, `_2` if strategy names collide).

### 3.3. Account & Position State Tracker (`portfolio/account.py`)

*   **`Position` Class**:
    *   Track active positions with fields: `ticker`, `quantity` (positive for Long, negative for Short), `avg_price`, `current_price`, and `multiplier`.
    *   Expose properties: `direction` ('LONG', 'SHORT', or 'FLAT'), `unrealized_pnl` (handles signs for Long/Short automatically since Short quantity is negative), and `market_value` (gross exposure: absolute value of quantity * price * multiplier).
    *   Implement average cost accounting inside `update_position(fill_qty, fill_price)` to update position size and average price when adding to a position, and compute realized PnL when reducing or reversing a position.
*   **`Account` Class**:
    *   Track balances: `balance` (realized cash), `equity` (balance + unrealized PnL), `used_margin` (exposure * margin requirement), `free_margin` (equity - used margin), `margin_level` ((equity / used_margin) * 100), and `is_margin_called` flag (equity <= used_margin).
    *   Implement `apply_fill(ticker, action, quantity, price, commission)` to update position and account balances. Removes flat/closed positions from the active position map to conserve memory.

### 3.4. Event-Driven Shared Capital Simulation Engine (`portfolio/shared_engine.py`)

*   Simulates multiple strategies sharing a single account balance:
    *   Process sorted prioritized events chronologically.
    *   Before executing a `MarginRequestEvent`, validate that `available_margin >= required_margin` (where `required_margin = position_value / leverage`).
    *   **Compounding Sizing**: Dynamically scale each strategy's trade sizes. If a strategy specifies `initial_capital`, calculate its standalone unscaled equity over time. The scale factor applied to the trade is `(portfolio_equity * strategy_weight) / strategy_equity`. Otherwise, fall back to scaling by `strategy_weight`.
    *   **Capital Conflict Resolution**: If multiple strategies request capital at the *exact same time* and available margin is insufficient to fulfill all requests, execute as many as possible. Block/skip the remainder, logging the conflict timestamp, strategy name, margin requested, winner/loser status, and skipped trade details.
    *   **Stress Test Mode (MAE)**: Support a `--stress-test-drawdown` mode where floating PnL is modeled using the trade's Maximum Adverse Excursion (MAE) to simulate the worst-case drawdown path, rather than linear interpolation between entry and exit.
    *   **Simulation Metrics Output**: Return a results dict containing:
        *   Ending equity, CAGR, max drawdown %, and structured equity/drawdown curves.
        *   Monthly return percentage matrix.
        *   Granular trade statistics (executed vs. skipped count, win rate, profit factor, expectancy, average/max win and loss).
        *   Conflict report metrics: conflict frequency, skipped profit opportunity, avoided losses, average margin usage, peak margin usage, average concurrent positions, and capital/margin efficiency ratios (including time-weighted versions).

### 3.5. Position Sizing & Limits Risk Engine (`portfolio/risk.py`)

*   **`RiskParameters` Dataclass**:
    *   Define configurations for sizing modes (`fixed_capital`, `fixed_qty`, `risk_per_trade`, `kelly`, `vol_target`, `atr`) and limit parameters.
*   **Position Sizing**:
    *   Implement `calculate_position_size(equity, entry_price, ...)`:
        *   `risk_per_trade`: size = (equity * risk_pct) / stop_loss_distance.
        *   `kelly`: size = (equity * kelly_fraction * kelly_percentage) / entry_price.
        *   `vol_target`: weight = target_vol_ann / asset_vol_ann, size = (equity * weight) / entry_price.
        *   `atr`: size = (equity * risk_pct) / (atr * atr_multiplier).
*   **Risk Limits Enforcement**:
    *   Implement `validate_trade_entry(ticker, quantity, price, equity, active_positions_count, current_margin_used, current_portfolio_risk, timestamp, ...)` to validate pre-trade boundaries:
        *   **Max Drawdown Stop**: Halt all new entries if peak-to-trough drawdown exceeds `max_drawdown_pct`.
        *   **Max Daily Loss Limit**: Halt trading for the day if daily realized losses exceed `max_daily_loss_pct` of daily starting equity.
        *   **Max Concurrent Positions**: Skip entry if open positions equal or exceed `max_concurrent_positions`.
        *   **Max Margin Usage**: Skip entry if estimated margin usage exceeds `max_margin_usage_pct` of equity.
        *   **Max Portfolio Risk**: Skip entry if the sum of stop-loss risks (position size * stop distance) exceeds `max_portfolio_risk_pct` of equity.
    *   Return a consolidated `RiskReport` documenting evaluation details and blocked trade reasons.

### 3.6. SLSQP Portfolio Optimizer & Bootstrap Resampler (`portfolio/optimizer.py`)

*   **Base Curve Reconstruction**:
    *   To optimize weights, run each strategy independently with 100% capital to reconstruct a daily equity and cash metrics dataframe. Reindex and align all strategy dataframes to a unified daily timeline.
*   **SLSQP Optimization Engine**:
    *   Use `scipy.optimize.minimize(method="SLSQP")` to solve for the optimal strategy weights vector $w$.
    *   Apply bounds: $min\_weight \le w_i \le max\_weight$.
    *   Apply constraints: sum of weights $\le leverage$, minimum cash reserve, maximum portfolio risk (annualized volatility of portfolio returns), and maximum concurrent positions.
    *   Objective targets: `sharpe`, `drawdown` (minimizes maximum drawdown), `cagr`, `calmar`, `recovery_factor`, and `cvar`.
*   **Monte Carlo CVaR Bootstrap Resampler**:
    *   To optimize for `cvar` or calculate tail risk, implement a vectorized resampler.
    *   Given portfolio weights $w$, compute the daily returns vector. Shuffling this daily returns vector with replacement (bootstrap) to generate $N$ path paths (default 1,000 paths).
    *   Reconstruct cumulative equity paths, calculate the max drawdown for each path, and compute CVaR as the mean of the worst $(1 - confidence)$ portion of drawdown outcomes (e.g. worst 5% tail risk).
*   **Frontier Generation**:
    *   Implement `generate_efficient_frontier` which solves for minimum drawdown weights across a grid of target CAGR values.
    *   Implement `plot_efficient_frontier` saving the curve and highlighting the optimal portfolio point.

### 3.7. Analytics & Visualizations Engine (`portfolio/analytics.py`)

*   **Performance Metrics**:
    *   Resample raw equity curve into regular daily intervals. Compute daily returns.
    *   Implement equations for: CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown %, Max Drawdown Cash, Calmar/MAR Ratio, Recovery Factor, and the Ulcer Index.
    *   Process trade lists to output win rate, profit factor, expectancy, average win/loss, and consecutive win/loss streaks.
*   **Charts Generator (Matplotlib & Seaborn)**:
    *   Create clean, high-DPI visual charts:
        *   **Equity Curve**: Shaded growth line.
        *   **Drawdown Curve**: Filled area chart showing peak-to-trough losses.
        *   **Monthly Heatmap**: Diverging colored matrix displaying monthly returns.
        *   **Yearly Returns**: Bar chart with value annotations.
        *   **Return Distribution**: KDE density curve of daily returns.
        *   **Daily/Trade Histograms**: Standard frequency bins.
*   **Dashboard Compilation**:
    *   Generate a unified 3x3 dashboard figure displaying the performance metrics panel, equity curve, drawdown curve, return distribution, yearly returns, monthly return heatmap, and rolling window metrics.

### 3.8. Report Compiler (`portfolio/report.py`)

*   **PDF Compiler (ReportLab)**:
    *   Build a print-ready document template (`SimpleDocTemplate`).
    *   Implement a two-pass `NumberedCanvas` to calculate total pages dynamically. Draw running headers ("PORTFOLIO BACKTEST PERFORMANCE REPORT") and running footers ("Page X of Y") separated by thin slate dividers.
    *   Structure the report with: Title block, Executive Summary table, monthly returns matrix, trade statistics table, capital usage metrics, conflict logs summary, and risk limits report. Embed the generated charts. Use slate color schemes (`#0f172a`, `#475569`, `#e2e8f0`).
*   **Excel Compiler (Openpyxl)**:
    *   Generate a multi-sheet workbook containing tabs:
        *   `Dashboard`: Key metrics formatted in tables.
        *   `Monthly Returns`: Year-by-month grid matrix.
        *   `Trades List`: Detailed log of all simulated/scaled trades.
        *   `Margin Conflicts`: Log of skipped trades and avoided losses.
    *   Format tables with dark headers, alternating row colors, proper numeric/percentage formatting, and auto-adjusted column widths.
*   **CSV compiler**: Export flat tabular files for summaries, trades, and conflicts.

### 3.9. Orchestration CLI Entrypoint (`main.py`)

*   Use `argparse` to handle command line arguments for initial capital, leverage, folder path, output directory, optimization flags, constraints, risk limit parameters, stress-test flags, and CVaR details.
*   Implement execution workflow:
    1. Import and parse all TradingView Excel files in the target folder.
    2. If `--optimize` is enabled, run `PortfolioOptimizer` to solve for weights, plot the Efficient Frontier, and verify metrics by running a scaled simulation.
    3. If baseline, run `SharedCapitalEngine` directly.
    4. Compile PDF, Excel, and CSV reports, and write individual chart PNG files to the output directory.
    5. Output a structured, well-aligned performance summary to stdout.

---

## 4. Verification and Testing Directives

Write unit and integration tests under `tests/` using `pytest`. Implement the following tests:
1.  **importer tests**: Verify sheet/column resolution, and ensure multi-row TradingView trades are correctly merged.
2.  **event queue tests**: Confirm events sort chronologically and that tie-breaking correctly orders `MarginReleaseEvent` before `MarginRequestEvent`.
3.  **account state tests**: Verify LONG/SHORT position updates, average cost updates on adding contracts, and margin call conditions.
4.  **shared engine tests**: Validate that trades are skipped when available margin is insufficient, compounding scales correctly, conflict logs accurately track winners/losers, and MAE stress testing correctly calculates floating PnL.
5.  **risk engine tests**: Verify Kelly, Vol Target, and ATR position sizing calculations. Check that entries are blocked by daily loss limits, concurrent position counts, and drawdown stops.
6.  **optimizer tests**: Verify SLSQP constraint convergence (weights summing to leverage limit, min cash reserve enforcement) and Monte Carlo bootstrap CVaR calculations.
7.  **analytics and report tests**: Verify correct calculation of CAGR, Sharpe, Sortino, and Ulcer Index formulas, and check that reports compile without raising errors.
