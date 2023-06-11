import numpy as np

def AssetPaths(S0, mu, sigma, T, NSteps, NRepl):
    SPaths = np.zeros((NRepl, NSteps+1))
    SPaths[:, 0] = S0
    dt = T/NSteps
    nudt = (mu-0.5*sigma**2)*dt
    sidt = sigma*np.sqrt(dt)
    for i in range(NRepl):
        for j in range(NSteps):
            SPaths[i, j+1] = SPaths[i, j]*np.exp(nudt + sidt*np.random.normal())
    return SPaths