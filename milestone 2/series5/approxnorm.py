import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def approxnorm(n):
    approx_1 = np.sum(np.random.rand(n, 12), axis=1) - 6
    approx_2 = np.random.randn(n)

    plt.figure(1)
    stats.probplot(approx_1, dist="norm", plot=plt)
    plt.title('CLT approximation')

    plt.figure(2)
    stats.probplot(approx_2, dist="norm", plot=plt)
    plt.title('randn sample')

    plt.figure(3)
    plt.hist(approx_1, bins=np.arange(-5, 5, 0.1), alpha=1)
    plt.title('CLT approximation')

    plt.figure(4)
    plt.hist(approx_2, bins=np.arange(-5, 5, 0.1), alpha=1)
    plt.title('randn sample')

    plt.show()