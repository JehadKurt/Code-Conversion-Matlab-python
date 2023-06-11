import numpy as np
import math
from scipy.stats import norm

def DOPutMCCondIS(S0, K, r, T, sigma, Sb, NSteps, NRepl, bp):
    dt = T / NSteps
    nudt = (r - 0.5 * sigma ** 2) * dt
    b = bp * nudt
    sidt = sigma * math.sqrt(dt)
    Call, Put = blsprice(S0, K, r, T, sigma)
    NCrossed = 0
    Payoff = np.zeros(NRepl)
    Times = np.zeros(NRepl)
    StockVals = np.zeros(NRepl)
    ISRatio = np.zeros(NRepl)
    for i in range(NRepl):
        vetZ = nudt - b + sidt * np.random.randn(NSteps)
        LogPath = np.cumsum(np.concatenate(([np.log(S0)], vetZ)))
        Path = np.exp(LogPath)
        jcrossed = None
        indices = np.where(Path <= Sb)[0]
        if len(indices) > 0:
            jcrossed = np.min(indices)
        if jcrossed is not None:
            NCrossed += 1
            TBreach = jcrossed - 1
            Times[NCrossed - 1] = TBreach * dt
            StockVals[NCrossed - 1] = Path[jcrossed]
            ISRatio[NCrossed - 1] = np.exp(
                TBreach * b ** 2 / 2 / sigma ** 2 / dt + 
                b / sigma ** 2 / dt * np.sum(vetZ[0:TBreach]) -
                TBreach * b / sigma ** 2 * (r - sigma ** 2 / 2)
            )
    if NCrossed > 0:
        Caux, Paux = blsprice(StockVals[0:NCrossed], K, r,
                              T - Times[0:NCrossed], sigma)
        Payoff[0:NCrossed] = np.exp(-r * Times[0:NCrossed]) * Paux * ISRatio[0:NCrossed]
    Pdo, CI = norm.fit(Put - Payoff)
    return Pdo, CI, NCrossed
def blsprice(S, K, r, T, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    Put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return Call, Put
