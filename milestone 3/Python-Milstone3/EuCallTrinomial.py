import numpy as np

def EuCallTrinomial(S0, K, r, T, sigma, N, deltaX):
    # Precompute invariant quantities
    deltaT = T/N
    nu = r - 0.5*sigma**2
    discount = np.exp(-r*deltaT)
    p_u = discount*0.5*((sigma**2*deltaT+nu**2*deltaT**2)/deltaX**2 + nu*deltaT/deltaX)
    p_m = discount*(1 - (sigma**2*deltaT+nu**2*deltaT**2)/deltaX**2)
    p_d = discount*0.5*((sigma**2*deltaT+nu**2*deltaT**2)/deltaX**2 - nu*deltaT/deltaX)
    
    # set up S values (at maturity)
    Svals = np.zeros(2*N+1)
    Svals[0] = S0*np.exp(-N*deltaX)
    for j in range(1, 2*N+1):
        Svals[j] = np.exp(deltaX)*Svals[j-1]
    
    # set up lattice and terminal values
    Cvals = np.zeros((2*N+1, 2))
    t = N % 2
    for j in range(2*N+1):
        Cvals[j,t] = max(Svals[j]-K, 0)
    
    for t in range(N-1, -1, -1):
        know = t % 2
        knext = (t+1) % 2
        for j in range(N-t, N+t+1):
            Cvals[j,know] = p_d*Cvals[j-1,knext] + p_m*Cvals[j,knext] + p_u*Cvals[j+1,knext]
    
    price = Cvals[N,0]
    return price