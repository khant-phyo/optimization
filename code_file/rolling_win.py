import numpy as np
import matplotlib.pyplot as plt

def rolling_backtest_with_plots(
    returns,
    optimizer_fn,
    train_size=252,
    test_size=21,
    step_size=21,
    optimizer_name="Model",
    **optimizer_kwargs
):

    T, N = returns.shape

    all_returns = []
    all_weights = []
    sharpe_per_window = []

    for start in range(0, T - train_size - test_size, step_size):
        train = returns[start : start + train_size]

        mu = train.mean(axis=0)
        Sigma = np.cov(train, rowvar=False)

        test = returns[start + train_size : start + train_size + test_size]

        if len(all_weights) == 0:
            w_prev = np.ones(N) / N
        else:
            w_prev = all_weights[-1]

        w_opt, _ = optimizer_fn(
            mu, Sigma, w_prev,
            **optimizer_kwargs
        )

        all_weights.append(w_opt.copy())

        port_ret = test @ w_opt
        all_returns.extend(port_ret)

        if np.std(port_ret) > 0:
            sharpe = np.mean(port_ret) / np.std(port_ret) * np.sqrt(252)
        else:
            sharpe = 0

        sharpe_per_window.append(sharpe)

        print(f"{optimizer_name} Window {start}: Sharpe = {sharpe:.4f}")
        
        # print(np.linalg.norm(all_weights[-1] - all_weights[-2]))
    all_returns = np.array(all_returns)

    cumulative = np.cumprod(1 + all_returns)

    overall_sharpe = (
        np.mean(all_returns) /
        np.std(all_returns)
        * np.sqrt(252)
    )

    total_return = cumulative[-1] - 1

    print("overall_sharpe:", overall_sharpe)
    print("total_return:", total_return)

    # # 1. Equity curve
    # plt.figure(figsize=(10,4))
    # plt.plot(cumulative)
    # plt.title(f"{optimizer_name} - Portfolio Growth")
    # plt.xlabel("Time")
    # plt.ylabel("Wealth")
    # plt.show()

    # # 2. Sharpe per window
    # plt.figure(figsize=(8,4))
    # plt.plot(sharpe_per_window)
    # plt.title(f"{optimizer_name} - Sharpe per Window")
    # plt.xlabel("Window")
    # plt.ylabel("Sharpe")
    # plt.show()

    # # 3. Turnover
    # weight_changes = [
    #     np.linalg.norm(all_weights[i] - all_weights[i-1], ord=1)
    #     for i in range(1, len(all_weights))
    # ]

    # plt.figure(figsize=(8,4))
    # plt.plot(weight_changes)
    # plt.title(f"{optimizer_name} - Portfolio Turnover")
    # plt.xlabel("Window")
    # plt.ylabel("Weight Change (L1)")
    # plt.show()

    return {
        "cumulative": cumulative,
        "sharpe": overall_sharpe,
        "returns": all_returns,
        "weights": all_weights,
        "window_sharpe": sharpe_per_window
    }