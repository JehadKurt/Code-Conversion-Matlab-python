import numpy as np
import matplotlib.pyplot as plt

# Samples of a geometric Brownian motion at t=T are generated. The
# expectation E((K-S_1)_+) is approximated. Confidence bounds using CLT are
# computed.
def ex43c():
    
    M = 10**6
    T = 1
    S0 = 10
    sigma = 0.5  # volatility
    K = 11  # strike price
    X = np.sqrt(T) * np.random.randn(1, M)

    S = S0 * np.exp(sigma * X - 0.5 * T * (sigma**2))  # simulated stock prices
    HS = np.maximum(K - S, 0)  # calculation of payoff
    price = np.mean(HS)

    # 95% confidence intervals
    AM = price - 1.96 * np.sqrt(np.var(HS) / M)  # based on CLT
    BM = price + 1.96 * np.sqrt(np.var(HS) / M)  # %

    print('Estimated value:', price)
    print('Confidence interval:', [AM, BM])
    
    
def ex43a():
    t = 1
    N = 10**5
    M = 10
    c = 1
    sigma = 0.5
    x = np.concatenate(([np.zeros(M)], np.sqrt(t / N) * np.cumsum(np.random.randn(N, M), axis=0)), axis=0)
    drift = c * np.tile(np.arange(0, t + t / N, t / N)[:, np.newaxis], (1, M))
    x = sigma * x + drift
    x = np.exp(x)
    plt.plot(np.arange(0, t + t / N, t / N), np.column_stack((x, np.exp(drift[:, 0]))))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()