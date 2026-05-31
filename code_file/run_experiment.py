import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv("data_file/all_stocks_5yr.csv")

print(data_frame.head())
print(data_frame.columns)

#          date   open   high    low  close    volume Name
# 0  2013-02-08  15.07  15.12  14.63  14.75   8407500  AAL
# 1  2013-02-11  14.89  15.01  14.26  14.46   8882000  AAL
# 2  2013-02-12  14.45  14.51  14.10  14.27   8126000  AAL
# 3  2013-02-13  14.30  14.94  14.25  14.66  10259500  AAL
# 4  2013-02-14  14.94  14.96  13.16  13.99  31879900  AAL

# Index(['date', 'open', 'high', 'low', 'close', 'volume', 'Name'], dtype='str')

data_frame["date"] = pd.to_datetime(data_frame["date"])

print(data_frame["date"].min())
print(data_frame["date"].max())

# 2013-02-08 00:00:00
# 2018-02-07 00:00:00

prices = data_frame.pivot(
    index="date",
    columns="Name",
    values="close"
)

print(prices.head())
print(prices.columns)

# Name            A    AAL    AAP     AAPL   ABBV    ABC    ABT    ACN   ADBE  ...   XLNX    XOM   XRAY    XRX    XYL    YUM    ZBH   ZION    ZTS
# date                                                                         ...                                                               
# 2013-02-08  45.08  14.75  78.90  67.8542  36.25  46.89  34.41  73.31  39.12  ...  37.51  88.61  42.87  31.84  27.09  65.30  75.85  24.14  33.05
# 2013-02-11  44.60  14.46  78.39  68.5614  35.85  46.76  34.26  73.07  38.64  ...  37.46  88.28  42.84  31.96  27.46  64.55  75.65  24.21  33.26
# 2013-02-12  44.62  14.27  78.60  66.8428  35.42  46.96  34.30  73.37  38.89  ...  37.58  88.46  42.87  31.84  27.95  64.75  75.44  24.49  33.74
# 2013-02-13  44.75  14.66  78.97  66.7156  35.27  46.64  34.46  73.56  38.81  ...  37.80  88.67  43.08  32.00  28.26  64.41  76.00  24.74  33.55
# 2013-02-14  44.58  13.99  78.84  66.6556  36.57  46.77  34.70  73.13  38.61  ...  38.44  88.52  42.91  32.12  28.47  63.89  76.34  24.63  33.27

# [5 rows x 505 columns]
# Index(['A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABT', 'ACN', 'ADBE', 'ADI',
#        ...
#        'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS'],
#       dtype='str', name='Name', length=505)

prices = prices.dropna(axis=1)

log_return = np.log(prices/prices.shift(1)).dropna()

print(log_return.head())
print(log_return.columns)

mu = log_return.mean().values
Sigma = log_return.cov().values

print(log_return.shape[1])
# 470

N = log_return.shape[1]
weight_prev = np.ones(N) / N

from pgd_funs import pgd
from rolling_win import rolling_backtest_with_plots
from sim_annealing_funs import simulated_annealing
from diff_evolution_funs import differential_evolution

pgd_result = rolling_backtest_with_plots(
    returns=log_return.values,
    optimizer_fn=pgd,
    train_size=252,
    test_size=21,
    step_size=21,
    optimizer_name="PGD",
    lambder=0.5,
    gamma=0.01,
    learning_rate=0.01,
    no_steps=200
)

sa_result = rolling_backtest_with_plots(
    returns=log_return.values,
    optimizer_fn=simulated_annealing,
    train_size=252,
    test_size=21,
    step_size=21,
    optimizer_name="Simulated Annealing",
    lambder=0.5,
    gamma=0.01,
    T0=1.0,
    alpha=0.97,
    no_steps=300
)

de_result = rolling_backtest_with_plots(
    returns=log_return.values,
    optimizer_fn=differential_evolution,
    train_size=252,
    test_size=21,
    step_size=21,
    optimizer_name="Differential Evolution",
    lambder=0.5,
    gamma=0.01,
    population_size=20,
    F=0.7,
    CR=0.8,
    generations=80
)


def run_backtest_only(returns, optimizer_fn, **kwargs):

    T, N = returns.shape

    all_returns = []
    all_weights = []
    sharpe_per_window = []

    for start in range(0, T - 252 - 21, 21):

        train = returns[start:start+252]
        test = returns[start+252:start+252+21]

        mu = train.mean(axis=0)
        Sigma = np.cov(train, rowvar=False)

        weight_prev = np.ones(N) / N if len(all_weights) == 0 else all_weights[-1]

        w_opt, _ = optimizer_fn(mu, Sigma, weight_prev, **kwargs)

        all_weights.append(w_opt.copy())

        port_ret = test @ w_opt
        all_returns.extend(port_ret)

        if np.std(port_ret) > 0:
            sharpe = np.mean(port_ret) / np.std(port_ret) * np.sqrt(252)
        else:
            sharpe = 0

        sharpe_per_window.append(sharpe)

    return np.array(all_returns), np.array(all_weights), np.array(sharpe_per_window)


pgd_ret, pgd_w, pgd_sharpe = run_backtest_only(
    log_return.values,
    pgd,
    lambder=0.5,
    gamma=0.0001,
    learning_rate=0.1,
    no_steps=200
)

sa_ret, sa_w, sa_sharpe = run_backtest_only(
    log_return.values, simulated_annealing,
    lambder=0.5, gamma=0.0001,
    T0=1.0, alpha=0.995, no_steps=2000
)

de_ret, de_w, de_sharpe = run_backtest_only(
    log_return.values,
    differential_evolution,
    lambder=0.5,
    gamma=0.0001,
    population_size=20,
    F=0.9,
    CR=0.8,
    generations=80
)

min_len = min(len(pgd_ret), len(sa_ret), len(de_ret))

pgd_curve = np.cumprod(1 + pgd_ret[:min_len])
sa_curve  = np.cumprod(1 + sa_ret[:min_len])
de_curve  = np.cumprod(1 + de_ret[:min_len])

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))

# wealth curve
ax1.plot(pgd_curve, label="PGD", linewidth=1)
ax1.plot(sa_curve, label="SA", linewidth=1, color="red")
ax1.plot(de_curve, label="DE", linewidth=1)

ax1.set_title("Portfolio Growth Comparison")
ax1.set_xlabel("Time")
ax1.set_ylabel("Wealth")
ax1.grid(True)
ax1.legend()

# sharpe curve
ax2.plot(pgd_sharpe, label="PGD")
ax2.plot(sa_sharpe, label="SA", color="red")
ax2.plot(de_sharpe, label="DE")

ax2.set_title("Sharpe Ratio per Window")
ax2.set_xlabel("Window")
ax2.set_ylabel("Sharpe")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
