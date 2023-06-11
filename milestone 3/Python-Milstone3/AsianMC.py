from scipy.stats import norm
from AssetPaths import AssetPaths
import numpy as np
def AsianMC(S0, K, r, T, sigma, NSamples, NRepl):
    Payoff = np.zeros((NRepl, ))
    for i in range(NRepl):
        Path = AssetPaths(S0, r, sigma, T, NSamples, 1)
        Payoff[i] = max(0, np.mean(Path[0, 1:(NSamples+1)]) - K)
    P1 = np.mean(np.exp(-r*T)*Payoff)
    se = np.std(np.exp(-r*T)*Payoff, ddof=1)/np.sqrt(NRepl)
    alpha = 0.05
    CI1 = norm.interval(1-alpha, loc=P1, scale=se)
    return 'P1:', P1, 'CI:' , CI1
