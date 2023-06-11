import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from time import time
from brownp import brownp

def sde_convstrongmil():
    N0 = 10
    L = 6
    M = 10**4

    T = 1
    x0 = 1
    f = lambda x, t: -np.sin(x) * (np.cos(x))**3
    g = lambda x, t: np.cos(x)**2
    hfu = lambda x, t: 2 * np.cos(x)**3 * (-np.sin(x))

    NL = N0 * 2**L
    B = brownp(T, NL, M)
    YT = np.zeros((L+1, M))

    start_time = time()

    for l in range(L+1):
        N = N0 * 2**l
        p = 2**(L-l)
        h = T / N
        Y = np.zeros((N+1, M))
        Y[0, :] = x0

        for j in range(1, N+1):
            dB = B[j*p, :] - B[(j-1)*p, :]
            t = j * h
            x = Y[j-1, :]
            Y[j, :] = x + f(x, t) * h + g(x, t) * dB
            Y[j, :] = x + f(x, t) * h + g(x, t) * dB + 0.5 * hfu(x, t) * (dB**2 - h)

        YT[l, :] = Y[N, :]

    exact = np.arctan(B[-1, :] + np.tan(x0))
    YTe = np.abs(YT - np.tile(exact, (L+1, 1)))
    YTem = np.mean(YTe, axis=1)
    YTem2 = np.sqrt(np.sum(YTe**2, axis=1))

    Lp = L
    hv = T / (N * 2 ** np.arange(0, Lp+1))
    hL = hv[-1]
    p = hv[0] / hL

    plt.loglog(hv, YTem2, '-gx', hv, YTem, '-ro', hL * np.array([1, p]), YTem[-1] * np.array([1, p**1]), hL * np.array([1, p]), YTem2[-1] * np.array([1, p**1]))
    plt.legend(['strong error in L^1', 'strong error in L^2', 'h^{1}', 'h^{1}'])
    plt.grid(True)
    plt.xlabel('step size $h$')
    plt.ylabel('error')
    plt.show()

    slope, _, _, _, _ = linregress(np.log(hv), np.log(YTem))
    print('Strong rate of convergence:', slope)

    elapsed_time = time() - start_time
    print('Elapsed time:', elapsed_time)
sde_convstrongmil()