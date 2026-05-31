import numpy as np

def project(w, upper=0.05):
    w = np.maximum(w, 0)
    s = np.sum(w)
    if s == 0:
        return np.ones_like(w) / len(w)
    return w / s


def objective(weight, mhu, sigma, weight_prev, lambder, gamma):
    epsilon = 1e-8
    risk = weight.T @ sigma @ weight
    ret = weight.T @ mhu
    turnover = np.sum(np.sqrt((weight - weight_prev)**2 + epsilon))
    return lambder * risk - (1 - lambder) * ret + gamma * turnover


def differential_evolution( mhu, sigma, w_prev, lambder=0.5, gamma=0.01, population_size=20, F=0.7, CR=0.8, generations=100):

    N = len(mhu)

    base = np.random.dirichlet(np.ones(N), size=population_size - 1)
    base = np.array([project(ind) for ind in base])

    extra = project(np.random.dirichlet(np.ones(N)))
    pop = np.vstack([extra, base])

    scores = np.array([
        objective(ind, mhu, sigma, w_prev, lambder, gamma)
        for ind in pop
    ])

    best_idx = np.argmin(scores)
    best = pop[best_idx].copy()

    best_history = []

    for _ in range(generations):

        for i in range(population_size):

            idxs = [idx for idx in range(population_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            mutant = a + F * (b - c)
            mutant = project(mutant)

            cross = np.copy(pop[i])
            mask = np.random.rand(N) < CR
            cross[mask] = mutant[mask]
            cross = project(cross)

            score_cross = objective(cross, mhu, sigma, w_prev, lambder, gamma)

            if score_cross < scores[i]:
                pop[i] = cross
                scores[i] = score_cross

                if score_cross < scores[best_idx]:
                    best_idx = i
                    best = cross.copy()

        best_history.append(scores[best_idx])

    return best, best_history