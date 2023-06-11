import numpy as np
from scipy.stats import norm

def AsianMCGeoCV(S0, K, r, T, sigma, NSamples, NRepl, NPilot):
    # precompute quantities
    DF = np.exp(-r * T)
    GeoExact = GeometricAsian(S0, K, r, T, sigma, 0, NSamples)
    # pilot replications to set control parameter
    GeoPrices = np.zeros(NPilot)
    AriPrices = np.zeros(NPilot)
    for i in range(NPilot):
        Path = AssetPaths(S0, r, sigma, T, NSamples, 1)
        GeoPrices[i] = DF * max(0, np.power(np.prod(Path[1:(NSamples+1)]), 1/NSamples) - K)
        AriPrices[i] = DF * max(0, np.mean(Path[1:(NSamples+1)]) - K)
    MatCov = np.cov(GeoPrices, AriPrices)
    c = - MatCov[0, 1] / np.var(GeoPrices, ddof=1)
    # MC run
    ControlVars = np.zeros(NRepl)
    for i in range(NRepl):
        Path = AssetPaths(S0, r, sigma, T, NSamples, 1)
        GeoPrice = DF * max(0, np.power(np.prod(Path[1:(NSamples+1)]), 1/NSamples) - K)
        AriPrice = DF * max(0, np.mean(Path[1:(NSamples+1)]) - K)
        ControlVars[i] = AriPrice + c * (GeoPrice - GeoExact)
    P = np.mean(ControlVars)
    # two-sided confidence interval
    bootstrap_means = np.zeros(NRepl)
    for i in range(NRepl):
        resamples = np.random.choice(ControlVars, size=NRepl, replace=True)
        bootstrap_means[i] = np.mean(resamples)
    ci_lower, ci_upper = np.percentile(bootstrap_means, [5, 95])
    CI = (ci_lower, ci_upper)
    return "P3:", P, "CI3:",  CI
    
def GeometricAsian(S0, K, r, T, sigma, delta, NSamples):
    dT = T / NSamples
    nu = r - sigma**2/2 - delta
    a = np.log(S0) + nu * dT * NSamples + 0.5 * nu * (T - dT * NSamples)
    b = sigma**2 * dT * NSamples * (NSamples + 1) * (2 * NSamples + 1) / 6 / NSamples**2
    x = (a - np.log(K) + b) / np.sqrt(b)
    P = np.exp(-r * T) * (np.exp(a + b/1.372) * norm.cdf(x) - K * norm.cdf(x - np.sqrt(b)))
    return P


def AssetPaths(S0, r, sigma, T, NSamples, NRepl):
    dt = T / NSamples
    nu = r - 0.5 * sigma**2
    S = np.zeros((NSamples + 1, NRepl))
    S[0] = S0
    for t in range(1, NSamples + 1):
        S[t] = S[t-1] * np.exp(nu * dt + sigma * np.sqrt(dt) * np.random.randn(NRepl))
    return S
