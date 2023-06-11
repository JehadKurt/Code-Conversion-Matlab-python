import numpy as np
from scipy.stats import norm

def DOPutMC(S0, K, r, T, sigma, Sb, NSteps, NRepl):
    np.random.seed(0)
    Call, Put = blsprice(S0, K, r, T, sigma)
    Payoff = np.zeros(NRepl)
    NCrossed = 0
    for i in range(NRepl):
        Path = AssetPaths1(S0, r, sigma, T, NSteps, 1)
        crossed = np.any(Path <= Sb)
        if not crossed:
            Payoff[i] = max(0, K - Path[NSteps])
        else:
            Payoff[i] = 0
            NCrossed += 1
    P, CI = norm.fit(np.exp(-r*T) * Payoff)
    return  P, CI 

def AssetPaths1(S0, r, sigma, T, NSteps, NRepl):
    dt = T / NSteps
    S = np.zeros((NSteps+1, NRepl))
    S[0] = S0
    for t in range(1, NSteps+1):
        z = np.random.normal(size=NRepl)
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return S

def blsprice(S0, K, r, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Call = S0 * norm.cdf(d1) - K*np.exp(-r*T) * norm.cdf(d2)
    Put = K*np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return Call, Put