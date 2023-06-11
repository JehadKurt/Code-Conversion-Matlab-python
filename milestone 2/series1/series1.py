import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
import numpy as np
import math
import matplotlib.pyplot as plt


def mcc1d(M, a, flag):
    '''
    Monte-Carlo method to compute the first two moments of a random variable Y=U^{-a},
    where U~Unif([0,1]) and a∈(−1,1).
    
    Parameters:
    M (int): number of samples
    a (float): coefficient in x^(-a)
    flag (int): 1 for CLT bounds, 2 for Chebyshev bounds with q=2, 3 for Chebyshev bounds with q=1/a
    
    Returns:
    y (ndarray): array of shape (2, M), where y[0, :] measures the error and y[1, :] is for confidence level
    '''
    q = 1 / a
    y = np.zeros((2, M))
    exact = 1 / (1 - a)
    delta = 0.05
    
    # Generate M samples of U
    Z = np.random.rand(M)
    
    # Compute the corresponding samples of Y
    X = 1 / Z**a
    
    # Compute sample mean and sample variance
    mean = np.cumsum(X) / np.arange(1, M + 1)
    varest = np.cumsum((X - mean)**2) / np.arange(1, M + 1)
    
    # Compute qth moment for Chebyshev bounds with q=1/a
    qmom = np.cumsum(np.abs(X - mean)**q) / np.arange(1, M + 1)
    
    # Compute absolute error from exact value
    err = np.abs(np.cumsum(X) / np.arange(1, M + 1) - exact)
    
    if flag == 1:
        # Confidence interval (CLT bounds)
        am = mean - erfinv(1 - delta) * np.sqrt(2) * np.sqrt(varest / np.arange(1, M + 1))
        bm = mean + erfinv(1 - delta) * np.sqrt(2) * np.sqrt(varest / np.arange(1, M + 1))
    elif flag == 2:
        # Confidence interval (Chebyshev bounds with q=2)
        am = mean - delta**(-1/2) * np.sqrt(varest / np.arange(1, M + 1))
        bm = mean + delta**(-1/2) * np.sqrt(varest / np.arange(1, M + 1))
    elif flag == 3:
        # Confidence interval (Chebyshev bounds with q=1/a)
        am = mean - delta**(-1/q) * qmom**(1/q) / np.arange(1, M + 1)**(1 - 1/q)
        bm = mean + delta**(-1/q) * qmom**(1/q) / np.arange(1, M + 1)**(1 - 1/q)
    else:
        print('Something is wrong!!')
        
    # Counts whether the exact mean is in [am, bm]
    y[0, :] = (bm > exact) * (am < exact)
    y[1, :] = err
    
    return y






def conflevel1(M, a):
    flag = 2
    Np = 10
    delta = 0.05
    q = 1 / a

    # Mean
    exact = 1 / (1 - a)

    # Variance
    exact2 = 1 / (1 - 2 * a) - exact ** 2
    if exact2 < 0:
        print('Variance does not exist!!')

    Z = np.random.rand(M)
    X = 1 / Z ** a

    mean = np.cumsum(X) / np.arange(1, M+1)
    varest = np.cumsum((X - mean) ** 2) / np.arange(1, M+1)
    qmom = np.cumsum(np.abs(X - mean) ** q) / np.arange(1, M+1)
    err = np.abs(np.cumsum(X) / np.arange(1, M+1) - exact)

    if flag == 1:
        # Conf Inter. (CLT)
        am = mean - math.erfinv(1 - delta) * math.sqrt(2) * np.sqrt(varest / np.arange(1, M+1))
        bm = mean + math.erfinv(1 - delta) * math.sqrt(2) * np.sqrt(varest / np.arange(1, M+1))
    elif flag == 2:
        # Conf Inter. (Chebyshev)
        am = mean - delta ** (-0.5) * np.sqrt(varest / np.arange(1, M+1))
        bm = mean + delta ** (-0.5) * np.sqrt(varest / np.arange(1, M+1))
    elif flag == 3:
        # Conf Inter. (Chebyshev) qth moment
        am = mean - delta ** (-1 / q) * qmom ** (1 / q) / np.arange(1, M+1) ** (1 - 1 / q)
        bm = mean + delta ** (-1 / q) * qmom ** (1 / q) / np.arange(1, M+1) ** (1 - 1 / q)
    else:
        print('Something is wrong!!')
        return

    errp = err[np.arange(Np-1, M, Np)]  # plot only every Np-th point
    errvp = np.abs(varest[np.arange(Np-1, M, Np)] - exact2)
    varest = varest[np.arange(Np-1, M, Np)]
    amp = am[np.arange(Np-1, M, Np)]
    bmp = bm[np.arange(Np-1, M, Np)]
    meanp = mean[np.arange(Np-1, M, Np)]
    Mp = np.arange(Np, M+1, Np)

    M1 = Mp[0]
    M2 = Mp[-1]




       # Subplot 1
    plt.subplot(3, 1, 1)
    if len(errp) > 0:
        plt.loglog(Mp, errp, 'b', [M1, M2], [errp[-1]] * 2, 'r--', [M1, M2], errp[-1] * np.asarray([(M1/M2)**-0.5, 1]), 'g--')
        plt.legend(['|mean-mean_m|', 'constant', 'm^{-0.5}', 'm^{-1/3}'])
    else:
        plt.loglog(Mp, errp, 'b')
        plt.legend(['|mean-mean_m|'])
    plt.title('Convergence of empirical moments')
    plt.ylabel('Error')
    # Subplot 2
    plt.subplot(3, 1, 2)
    plt.plot(Mp, meanp, 'r', Mp, varest, 'g', Mp, amp, 'b', Mp, bmp, 'b')
    plt.legend(['Mean', 'Variance', 'Lower bound', 'Upper bound'])

    # Subplot 3
    # Subplot 3
    plt.subplot(3, 1, 3)
    if len(errvp) > 0:
        plt.loglog(Mp, errvp, 'b', [M1, M2], [errvp[-1]] * 2, 'r--', [M1, M2], errvp[-1] * np.array([(M1/M2)**-0.5, 1]), 'g--')
        plt.legend(['|var-var_m|', 'constant', 'm^{-0.5}', 'm^{-1/3}'])


    # Display the plots
    plt.show()



def conflevel(a, M, K):
    '''
    Computes the confidence level for Monte Carlo simulation with mcc1d function, 
    using either CLT bounds or Chebyshev bounds with q=2 or q=1/a.
    
    Parameters:
   
    Returns:
    None
    '''
    flag = 1
    import numpy as np
    y = np.zeros((2, M)) # y[0, :] measures the error, y[1, :] is for confidence level

    if a > 0.5:
        print('Variance does not exist!!')

    for k in range(K):
        y += mcc1d(M, a, flag)

    Np = 10
    import numpy as np
    Mp = np.arange(Np, M + 1, Np)
    yp = y[:, Np - 1::Np] / K

    M1, M2 = Mp[0], Mp[-1]

    import numpy as np

  
    plt.subplot(2, 1, 1)
    plt.loglog(Mp, yp[1, :], 'b', [M1, M2], yp[1, -1] * np.array([(M1 / M2)**-0.5, 1]), 'r',
               [M1, M2], yp[1, -1] * np.array([(M1 / M2)**-(1/3), 1]), 'g')
    plt.legend(['|mu-mu_m|', 'm^{-0.5}', 'm^{-1/3}'])

    plt.subplot(2, 1, 2)
    plt.plot(Mp, yp[0, :])
    plt.legend(['Confidence level'])
    

