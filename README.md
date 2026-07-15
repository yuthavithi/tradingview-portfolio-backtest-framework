# TradingView Portfolio Backtest & Optimization Framework

A professional, event-driven portfolio simulation and optimization framework designed to backtest multiple TradingView strategies sharing a single capital account. It models capital efficiency, margin requirements, execution conflicts, risk limits, and uses Monte Carlo Bootstrap Resampling for Conditional Value at Risk (CVaR) optimization.

---

## Key Features

*   **Excel Importer**: Parses exported TradingView XLSX strategy reports (including trades, orders, entry/exit prices, contracts, and Maximum Adverse Excursion).
*   **Event-Driven Shared Capital Simulation**: Simulates multiple strategies running concurrently under a shared capital pool. Supports:
    *   **Compounding**: Dynamic position scaling based on growing or shrinking account equity.
    *   **Margin & Leverage Constraints**: Validates margin availability before trade execution.
    *   **Conflict Resolution**: Chronologically processes overlapping trades and records skipped profit or avoided losses.
*   **Stress Test Mode**: Models floating PnL using **Maximum Adverse Excursion (MAE)** to simulate worst-case adverse price movements rather than simple linear interpolation.
*   **Portfolio Optimizer**: Finds optimal strategy allocation weights using the SLSQP algorithm. Supported objectives:
    *   `sharpe`: Annualized daily Sharpe Ratio.
    *   `drawdown`: Minimizes maximum historical drawdown.
    *   `cagr`: Maximizes Compound Annual Growth Rate.
    *   `calmar`: CAGR divided by Max Drawdown.
    *   `recovery_factor`: Net Profit divided by Drawdown Cash.
    *   `cvar`: Minimizes **Monte Carlo Conditional Value at Risk** (Conditional Drawdown at Risk).
*   **Monte Carlo Bootstrap Resampling**: Shuffles historical daily portfolio returns (default: 1,000 iterations) to simulate parallel path alternatives, eliminating sequence risk and path-dependency to isolate true tail risk (worst 5% average drawdown).
*   **Dynamic Risk Engine**: Enforces portfolio-level rules (Max Daily Loss, Drawdown Halting, Max Margin Usage).
*   **Reporting & Visualization**: Generates:
    *   Professional PDF reports (with executive summary, performance matrix, charts, conflict log, risk limits).
    *   Multi-sheet Excel workbooks and flat CSV outputs.
    *   Plots: Equity/Drawdown curves, Monthly return heatmaps, Yearly returns, Capital usage, and the **Efficient Frontier**.

---

## Project Structure

```
├── main.py                     # CLI entrypoint and orchestrator
├── portfolio/
│   ├── importer.py             # TradingView Excel importer
│   ├── shared_engine.py        # Event-driven shared capital simulation engine
│   ├── optimizer.py            # Portfolio weight optimizer & MC CVaR calculator
│   ├── analytics.py            # Advanced performance & risk ratios calculator
│   ├── risk.py                 # Risk Engine & limits manager
│   ├── report.py               # PDF, Excel, and CSV report compiler
│   ├── events.py               # Margin events definitions
│   └── queue.py                # Priority event queue
├── tests/                      # Unit and integration test suite
└── README.md                   # Documentation
```

---

## Installation & Setup

1. **Prerequisites**: Python 3.10+
2. **Setup virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate    # macOS/Linux
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages include: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`, `reportlab`, `pypdf`.*

---

## Command Line Interface (CLI)

Run `main.py` from the root directory.

### Core CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--folder` | `str` | *Required* | Path to the directory containing TradingView Excel strategy reports. |
| `--output` | `str` | `output_reports` | Directory where output reports and charts will be saved. |
| `--capital` | `float` | `1000.0` | Initial account equity in USDT. |
| `--leverage` | `float` | `1.0` | Margin leverage factor (e.g. 10.0 for 10x). |
| `--stress-test-drawdown` | `flag` | `False` | Models floating PnL using Maximum Adverse Excursion (MAE). |
| `--disable-risk-limits`| `flag` | `False` | Disables the Risk Engine to show raw strategy performance. |
| `--optimize` | `flag` | `False` | Enables portfolio weight optimization across strategies. |
| `--objective` | `str` | `sharpe` | Optimization target: `sharpe`, `drawdown`, `cagr`, `calmar`, `recovery_factor`, `cvar`. |
| `--cvar-iterations` | `int` | `1000` | Number of bootstrap paths for MC CVaR calculation. |
| `--cvar-confidence` | `float` | `0.95` | Confidence level for tail risk CVaR (e.g. 0.95 for worst 5%). |
| `--min-weight` | `float` | `0.0` | Minimum allocation weight per strategy. |
| `--max-weight` | `float` | `1.0` | Maximum allocation weight per strategy. |
| `--min-cash` | `float` | `0.0` | Cash reserve constraint (fraction if <= 1.0, else absolute). |
| `--max-risk` | `float` | `None` | Annualized volatility limit constraint. |
| `--max-concurrent-positions` | `int` | `None` | Maximum allowed concurrent strategy positions constraint. |

---

## Usage Examples

### 1. Run Baseline Portfolio Backtest
Run a standard simulation where all strategies in the directory are run with 10x leverage and full compounding (each starting at 100% allocation):
```bash
python main.py --folder ./strategies_folder --output ./results_baseline --capital 1000 --leverage 10 --stress-test-drawdown --disable-risk-limits
```

### 2. Optimize Weights to Minimize Monte Carlo CVaR
Find the optimal strategy weights that minimize portfolio CVaR (worst-case drawdown risk under 1,000 bootstrap simulations) under a 10x leverage limit:
```bash
python main.py --folder ./strategies_folder --output ./results_cvar_optimized --capital 1000 --leverage 10 --optimize --objective cvar --cvar-iterations 1000 --cvar-confidence 0.95 --stress-test-drawdown --disable-risk-limits
```

### 3. Optimize Weights for Sharpe Ratio with Constraints
Optimize for maximum Sharpe Ratio while ensuring:
*   At least 15% cash reserve is maintained at all times.
*   Individual strategy weights are bounded between 10% and 60%.
*   Maximum concurrent positions across strategies is capped at 3.
```bash
python main.py --folder ./strategies_folder --output ./results_sharpe_constrained --capital 1000 --leverage 5 --optimize --objective sharpe --min-weight 0.1 --max-weight 0.6 --min-cash 0.15 --max-concurrent-positions 3
```

---

## Understanding the Outputs

All results are saved in the directory specified by `--output`:

*   **`portfolio_report.pdf`**: Comprehensive, print-ready PDF report including the Executive Summary, aligned Monthly Returns matrix, Trade stats, Capital usage stats, Conflict log summary, Risk engine limits, and **Monte Carlo CVaR stress test analysis**.
*   **`portfolio_report.xlsx`**: Excel workbook containing tabs for the Dashboard, aligned monthly returns, raw trades list, and margin conflicts logs.
*   **`equity_drawdown.png`**: Aligned equity growth and drawdown curve charts.
*   **`monthly_heatmap.png`**: Heatmap visualizing percentage returns by month.
*   **`efficient_frontier.png`**: Frontier curve illustrating the risk-return spectrum and highlighting the optimal portfolio weights.
