import numpy as np
from scipy.stats import norm
from AssetPaths import AssetPaths


def AsianMCCV(S0,K,r,T,sigma,NSamples,NRepl,NPilot):
    # pilot replications to set control parameter
    TryPath=AssetPaths(S0,r,sigma,T,NSamples,NPilot)
    StockSum = np.sum(TryPath,1)
    PP = np.mean(TryPath[:,1:(NSamples+1)],1)
    TryPayoff = np.exp(-r*T) * np.maximum(0, PP - K)
    MatCov = np.cov(StockSum, TryPayoff)
    c = - MatCov[0,1] / np.var(StockSum)
    dt = T / NSamples
    ExpSum = S0 * (1 - np.exp((NSamples + 1)*r*dt)) / (1 - np.exp(r*dt))
    
    # MC run with control variates and bootstrapping
    ControlVars = np.zeros(NRepl)
    for i in range(NRepl):
        StockPath = AssetPaths(S0,r,sigma,T,NSamples,1)
        Payoff = np.exp(-r*T) * np.maximum(0, np.mean(StockPath[:,1:(NSamples+1)],1) - K)
        ControlVars[i] = Payoff + c * (np.sum(StockPath) - ExpSum)
    
    # bootstrap for confidence interval
    resamples = np.random.choice(ControlVars, size=(NRepl, NRepl), replace=True)
    bootstrap_means = np.mean(resamples, axis=1)
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
    P2 = np.mean(ControlVars)
    CI2 = (ci_lower, ci_upper)
    return 'P2:', P2, "CI2:", CI2