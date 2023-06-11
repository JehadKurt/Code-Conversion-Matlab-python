import numpy as np
from scipy.stats import norm
from math import exp, sqrt
np.random.seed(0)
from AssetPaths import AssetPaths

def blsprice(S, K, r, T, sigma):
    if np.isscalar(S):
        S = np.array([S])
        K = np.array([K])
        T = np.array([T])
        sigma = np.array([sigma])
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    Put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return Call, Put

def DOPutMCCond(S0, K, r, T, sigma, Sb, NSteps, NRepl):
    dt = T/NSteps
    Call, Put = blsprice(S0, K, r, T, sigma)
    
    # Generate asset paths and payoffs for the down and in option
    NCrossed = 0
    Payoff = np.zeros(NRepl)
    Times = np.zeros(NRepl)
    StockVals = np.zeros(NRepl)
    for i in range(NRepl):
        Path = AssetPaths(S0, r, sigma, T, NSteps, 1)
        try:
            tcrossed = np.min(np.where(Path <= Sb)[1])
        except ValueError:
            tcrossed = np.nan
        if not np.isnan(tcrossed):
            NCrossed += 1
            Times[NCrossed-1] = (tcrossed-1) * dt
            StockVals[NCrossed-1] = Path[0,tcrossed]
    
    if NCrossed > 0:
        Caux, Paux = blsprice(StockVals[:NCrossed], K, r, T-Times[:NCrossed], sigma)
        Payoff[:NCrossed] = np.exp(-r*Times[:NCrossed]) * Paux
    
    Pdo, CI = norm.fit(Put - Payoff)
    return Pdo, CI, NCrossed
