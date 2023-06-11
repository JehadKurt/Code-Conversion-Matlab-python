import numpy as np
from scipy.stats import norm

def GeometricAsian(S0, K, r, T, sigma, delta, NSamples):
    dT = T / NSamples
    nu = r - sigma**2 / 2 - delta
    a = np.log(S0) + nu * dT + 0.5 * nu * (T - dT)
    b = sigma**2 * dT + sigma**2 * (T - dT) * (2 * NSamples - 1) / 6 / NSamples
    x = (a - np.log(K) + b) / np.sqrt(b)
    P = np.exp(-r * T) * (np.exp(a + b / 2) * norm.cdf(x) - K * norm.cdf(x - np.sqrt(b)))
    return P