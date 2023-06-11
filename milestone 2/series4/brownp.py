import numpy as np

def brownp(T, N, M=1):
    """
    Find M paths of Brownian motion, default M=1.
    Each column of x contains values at t=T/N, 2*T/N, ... , T.
    """
    x = np.zeros((N+1, M))
    x[1:] = np.sqrt(T/N) * np.cumsum(np.random.randn(N, M), axis=0)
    return x