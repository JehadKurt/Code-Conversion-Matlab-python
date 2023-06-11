import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def PlotBLS(S0, X, r , sigma):

    for T in np.arange(2, -0.25, -0.24999):
        plt.plot(S0, norm.cdf((np.log(S0/X) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))) * S0 
                 - norm.cdf((np.log(S0/X) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))) * X * np.exp(-r*T))

    plt.axis([30, 70, -5, 35])
    plt.grid(True)
    plt.show()