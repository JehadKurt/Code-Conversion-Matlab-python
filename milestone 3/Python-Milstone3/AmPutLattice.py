import numpy as np

def AmPutLattice(S0, K, r, T, sigma, N):
    # Precompute invariant quantities
    deltaT = T/N
    u = np.exp(sigma * np.sqrt(deltaT))
    d = 1/u
    p = (np.exp(r*deltaT) - d)/(u-d)
    discount = np.exp(-r*deltaT)
    p_u = discount*p
    p_d = discount*(1-p)
    
    # set up S values
    SVals = np.zeros(2*N+1)
    SVals[N] = S0
    for i in range(1, N+1):
        SVals[N+i] = u*SVals[N+i-1]
        SVals[N-i] = d*SVals[N-i+1]
    
    # set up terminal values
    PVals = np.zeros(2*N+1)
    for i in range(0, 2*N+1, 2):
        PVals[i] = max(K-SVals[i], 0)
    
    # work backwards
    for tau in range(1, N+1):
        for i in range(tau+1, 2*N-tau+1, 2):
            hold = p_u*PVals[i+1] + p_d*PVals[i-1]
            PVals[i] = max(hold, K-SVals[i])
    
    price = PVals[N]
    print(sigma)
    return round(price,4)