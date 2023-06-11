import numpy as np
import matplotlib.pyplot as plt
from time import time
import numpy as np
from brownp import brownp

def sde_heston2():
    N0 = 10
    L = 5
    M = 10**4
    T = 1
    x0 = 10
    v0 = 0.5
    extra = 3
    xi = 0.25
    theta = 0.5
    r = 0.05
    kappa = 2
    G = lambda x: np.maximum(11 - x, 0)

    start_time = time()

    Le = L + extra
    Ne = N0 * 2**Le
    BI = brownp(T, Ne, M)
    BII = brownp(T, Ne, M)
    YT = np.zeros((L + 2, M))

    Lv = np.arange(0, L + 1).tolist() + [Le]

    for l in range(L + 2):
        le = Lv[l]
        N = N0 * 2**le
        p = 2**(Le - le)
        h = T / N
        x = x0
        v = v0

        for j in range(1, N + 1):
            dBI = BI[j * p, :] - BI[(j - 1) * p, :]
            dBII = BII[j * p, :] - BII[(j - 1) * p, :]
            x = x + r * x * h + (abs(v)**0.5) * x * dBI
            v = v + kappa * (theta - v) * h + xi * (abs(v)**0.5) * dBII

        YT[l, :] = x

    Ys = G(YT)
    YTe = np.abs(YT[:L+1, :] - YT[L+1, :][None, :])
    YTem = np.mean(YTe, axis=1)
    YTem2 = np.sqrt(np.sum(YTe**2, axis=1))
    Ym = np.mean(Ys, axis=1)
    est_var = np.var(Ys[:L+1, :], axis=1, ddof=1)

    AM = Ym[:L+1] - 1.96 * np.sqrt(est_var / M)
    BM = Ym[:L+1] + 1.96 * np.sqrt(est_var / M)
    hv = T / (N0 * 2**np.arange(0, L+1))
    hL = hv[-1]
    p = hv[0] / hL

    print('CLT confidence interval')
    print(np.column_stack((AM, BM)))

    plt.loglog(hv, YTem2, '-gx', hv, YTem, '-ro', hL * np.array([1, p]), YTem[-1] * np.array([1, p**0.5]), hL * np.array([1, p]), YTem2[-1] * np.array([1, p**0.5]))
    plt.legend(['strong error in L^1', 'strong error in L^2', ' h^{1/2}', ' h^{1/2}'])
    plt.grid(True)
    plt.xlabel('step size $h$')
    plt.ylabel('error')
    plt.show()

    elapsed_time = time() - start_time
    print('Elapsed time:', elapsed_time)


sde_heston2()