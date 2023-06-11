import numpy as np
import matplotlib.pyplot as plt

def acceptreject(n):
    x = ar_randy(n)
    plt.hist(x, bins=np.arange(1, 11, 1), rwidth=.95, align='mid')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Discrete Acceptance Rejection')
    plt.xticks(np.arange(1, 11, 1))
    plt.show()

def ar_randy(n):
    p = np.array([0.11, 0.12, 0.09, 0.08, 0.12, 0.10, 0.09, 0.09, 0.10, 0.10])
    j = 0
    x = np.zeros(n)
    while j < n:
        y = discreterandu(1)
        u = np.random.rand()
        c = 1.2
        if u <= p[int(y) - 1] / (c * 0.10):  # Cast y to an integer
            x[j] = y
            j += 1
    return x

def discreterandu(n):
    return np.ceil(10 * np.random.rand(n))