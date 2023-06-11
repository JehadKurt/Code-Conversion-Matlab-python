import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from time import time
from brownp import brownp

def sde_convweak():
    N0 = 10
    L = 4
    M = 5 * 10**4
    itera = 10

    T = 1
    x0 = 1
    f = lambda x, t: -np.sin(x) * (np.cos(x))**3
    g = lambda x, t: np.cos(x)**2
    G = lambda x: np.maximum(x - 1.1, 0)

    NL = N0 * 2**L
    B = brownp(T, NL, M)
    YT = np.zeros((L+1, M))
    Erro1 = np.zeros((L+1, 1))
    Erro2 = np.zeros((L+1, 1))

    start_time = time()

    for it in range(itera):
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

            YT[l, :] = Y[N, :]

        Z = G(YT)

        Ym = np.mean(Z, axis=1)
        Ymex = np.mean(G(np.arctan(np.random.randn(1, int(1e7)) * np.sqrt(T) + np.tan(x0))))
        Yme = np.abs(Ym - Ymex)
        Erro1 += Yme.reshape(-1, 1)
        Erro2 += (Yme**2).reshape(-1, 1)

    Erro1 /= itera
    Erro2 = np.sqrt(Erro2 / itera)
    Lp = L

    hv = T / (N * 2 ** np.arange(0, Lp+1))
    hL = hv[-1]
    p = hv[0] / hL

    plt.loglog(hv, Erro1, '-gx', hv, Erro2, '-ro', hL * np.array([1, p]), Erro1[-1] * np.array([1, p**1.0]), hL * np.array([1, p]), Erro2[-1] * np.array([1, p**1.0]))
    plt.legend(['weak error in L^1', 'weak error in L^2', 'h^{1}', 'h^{1}'])
    plt.grid(True)
    plt.xlabel('step size $h$')
    plt.ylabel('error')
    plt.show()

    slope, _, _, _, _ = linregress(np.log(hv), np.log(Erro2.flatten()))
    print('Weak rate of convergence L^2:', slope)

    slope, _, _, _, _ = linregress(np.log(hv), np.log(Erro1.flatten()))
    print('Weak rate of convergence L^1:', slope)

    elapsed_time = time() - start_time
    print('Elapsed time:', elapsed_time)

sde_convweak()
