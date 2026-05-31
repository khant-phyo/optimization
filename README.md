# Portfolio Optimization — PGD, Simulated Annealing & Differential Evolution

Comparison of three portfolio optimization algorithms — Projected Gradient Descent (PGD), Simulated Annealing (SA), and Differential Evolution (DE) — on S&P 500 stock data using a rolling-window backtest.

---

## Dataset

**File:** `data_file/all_stocks_5yr.csv`

**Source:** [S&P 500 stock data on Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500)

| Field | Description |
|-------|-------------|
| `date` | Trading date |
| `open` | Opening price |
| `high` | Daily high |
| `low` | Daily low |
| `close` | Closing price ← used in this project |
| `volume` | Trade volume |
| `Name` | Stock ticker symbol |

- **Time period:** 2013-02-08 to 2018-02-07
- **Number of assets (after dropping NaNs):** 470

Place the CSV file at the following location before running:

```
data_file/all_stocks_5yr.csv
```

---

## Requirements

Python 3.8 or higher is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas==3.0.2
matplotlib==3.10.8
numpy==2.4.4
```

---

## Code Files

| File | Location | Description |
|------|----------|-------------|
| `run_experiment.py` | root | **Main file — run this** |
| `pgd_funs.py` | root | Projected Gradient Descent optimizer |
| `sim_annealing_funs.py` | root | Simulated Annealing optimizer |
| `diff_evolution_funs.py` | root | Differential Evolution optimizer |
| `rolling_win.py` | root | Rolling window backtest logic |

---

## How to Run

```bash
python run_experiment.py
```

This will:
1. Load and preprocess the dataset
2. Run all three optimizers using a rolling-window backtest
3. Print a summary table with annualised return, volatility, Sharpe ratio, max drawdown, turnover, and runtime
4. Save `results_summary.csv` and `results_plots.png`

---

## Project Structure

```
.
├── data_file/
│   └── all_stocks_5yr.csv
├── run_experiment.py
├── pgd_funs.py
├── sim_annealing_funs.py
├── diff_evolution_funs.py
├── rolling_win.py
└── requirements.txt
```
