import numpy as np
import matplotlib.pyplot as plt



def SmartEurLattice(S0, K, r, T, sigma, N):
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
    SVals[0] = S0*d**N
    for i in range(1, 2*N+1):
        SVals[i] = u*SVals[i-1]
        
    # set up terminal CALL values
    CVals = np.zeros(2*N+1)
    for i in range(0, 2*N+1, 2):
        CVals[i] = max(SVals[i]-K, 0)
    
    # work backwards
    for tau in range(N-1, -1, -1):
        for i in range(0, 2*tau+1):
            SVals[i] = d*SVals[i+1]
            CVals[i] = p_u*CVals[i+2] + p_d*CVals[i]
            CVals[i] = max(CVals[i], SVals[i]-K)
            
    return CVals[0]

def blsprice(S0, K, r, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    Nd1 = 0.5 + 0.5*np.math.erf(d1/np.sqrt(2))
    Nd2 = 0.5 + 0.5*np.math.erf(d2/np.sqrt(2))
    return S0*Nd1 - K*np.exp(-r*T)*Nd2
