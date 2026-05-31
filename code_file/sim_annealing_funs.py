import numpy as np

def objective(w, mu, Sigma, w_prev, lambder, gamma):
    eps = 1e-8
    risk = w.T @ Sigma @ w
    ret = w.T @ mu
    turnover = np.sum(np.sqrt((w - w_prev)**2 + eps))

    return lambder * risk - (1 - lambder) * ret + gamma * turnover


def project(w, upper=0.05):
    w = np.clip(w, 0, upper)
    s = np.sum(w)
    if s == 0:
        return np.ones_like(w) / len(w)
    return w / s


def get_neighbor(w, scale=0.08):
    noise = np.random.normal(0, scale, size=len(w))
    w_new = w + noise
    w_new = np.maximum(w_new, 0)
    return project(w_new)


def simulated_annealing(mu, Sigma, w_prev,
                         lambder=0.5,
                         gamma=0.01,
                         T0=5.0,
                         alpha=0.995,
                         no_steps=300):

    N = len(mu)

    w = np.random.dirichlet(np.ones(N))
    w = project(w)
    best_w = w.copy()

    best_score = objective(w, mu, Sigma, w_prev, lambder, gamma)

    T = T0

    for _ in range(no_steps):

        w_new = get_neighbor(w, scale=0.08)

        current_score = objective(w, mu, Sigma, w_prev, lambder, gamma)
        new_score = objective(w_new, mu, Sigma, w_prev, lambder, gamma)

        delta = new_score - current_score

        if delta < 0:
            accept = True
        else:
            accept = np.random.rand() < np.exp(-delta / (T + 1e-12))

        if accept:
            w = w_new

        score = objective(w, mu, Sigma, w_prev, lambder, gamma)

        if score < best_score:
            best_score = score
            best_w = w.copy()

        T *= alpha

    return best_w, []