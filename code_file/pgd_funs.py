import numpy as np

def project_weights(weight, upper_bound=0.05, iters=5):

    for i in range(iters):
        weight = np.clip(weight, 0, upper_bound)
        if np.sum(weight) == 0:
            weight = np.ones_like(weight) / len(weight)
        else:
            weight = weight / np.sum(weight)
    return weight

def calculate_gradient(weight, mhu, Sigma, weight_prev, lambder, gamma):
    epsilon = 1e-8
    risk_gradient = 2 * (Sigma @ weight)
    return_gradient = mhu
    diff = weight - weight_prev
    
    gradient_turnover = diff / np.sqrt(diff**2 + epsilon)
    
    gradient = lambder * risk_gradient - (1 - lambder) * return_gradient + gamma * gradient_turnover
    return gradient

def objective(weight, mhu, Sigma, weight_prev, lambder, gamma):
    epsilon = 1e-8
    risk = weight.T @ Sigma @ weight
    ret = weight.T @ mhu
    turnover = np.sum(np.sqrt((weight - weight_prev)**2 + epsilon))
    return lambder * risk - (1 - lambder) * ret + gamma * turnover

def pgd_step(weight, mhu, Sigma, weight_prev, lambder, gamma, learning_rate):
    gradient = calculate_gradient(weight, mhu, Sigma, weight_prev, lambder, gamma)
    weight = weight - learning_rate * gradient
    weight = project_weights(weight)
    return weight

def pgd(mhu, Sigma, weight_prev, lambder=0.5, gamma=0.01, learning_rate=0.01, no_steps=200):
    N = len(mhu)
    weight = np.ones(N) / N
    loss_history = []
    
    for i in range(no_steps):
        weight = pgd_step(weight, mhu, Sigma, weight_prev, lambder, gamma, learning_rate)
        loss_history.append(objective(weight, mhu, Sigma, weight_prev, lambder, gamma))
    
    return weight, loss_history