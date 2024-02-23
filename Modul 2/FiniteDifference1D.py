import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple
import scipy.sparse as sps

def CDS_fd(Pe, N, dx, BV: Tuple[float, float]):
    s = np.zeros(N)
    s[0] = BV[0]
    s[-1] = BV[1]
    P = Pe*dx
    aw = -P/2 - 1
    ai = 2
    ae = P/2 - 1

    Aw = np.full(N-1,aw)
    Ai = np.full(N,ai)
    Ae = np.full(N-1,ae)

    Ai[0] = 1
    Ai[-1] = 1
    Ae[0] = 0
    Aw[-1] = 0

    data = np.array([Ai, Aw, Ae])
    D = sps.diags(data,[0,-1,1])

    return sps.linalg.spsolve(D, s)

def UDS_fd(Pe, N, dx, BV: Tuple[float, float]):
    s = np.zeros(N)
    s[0] = BV[0]
    s[-1] = BV[1]
    P = Pe*dx
    aw = - 1 - P
    ai = 2 + P
    ae = - 1

    Aw = np.full(N-1,aw)
    Ai = np.full(N,ai)
    Ae = np.full(N-1,ae)

    Ai[0] = 1
    Ai[-1] = 1
    Ae[0] = 0
    Aw[-1] = 0

    data = np.array([Ai, Aw, Ae])
    D = sps.diags(data,[0,-1,1])

    return sps.linalg.spsolve(D, s)

if __name__ == "__main__":
    P = [1, 1.5, 2, 5]
    L = 1
    N = 10
    # BVP parameters
    BV = (0, 1)
    dx = L/(N-1)
    error_CDS = np.zeros(len(P))
    error_UDS = np.zeros(len(P))
    for i, _P in enumerate(P):
        Pe = _P/dx
        T_CDS = CDS_fd(Pe, N, dx, BV)
        T_UDS = UDS_fd(Pe, N, dx, BV)

        fun = lambda x: (np.exp(Pe*x)-1)/(np.exp(Pe)-1)
        x = np.linspace(0, L, N)
        T_true = fun(x)

        error_CDS[i] = np.linalg.norm(T_true - T_CDS, np.inf)
        error_UDS[i] = np.linalg.norm(T_true - T_UDS, np.inf)

        # Plot the results in a single figure
        x_plot = np.linspace(0, L, 1000)
        T_true_plot = fun(x_plot)
        plt.figure()
        plt.suptitle("Test for Pe = {}".format(Pe))
        plt.plot(x_plot, T_true_plot, label="T_true")
        plt.plot(x, T_CDS, label="T_CDS", marker="o")
        plt.plot(x, T_UDS, label="T_UDS", marker="o")
        plt.xlabel("x")
        plt.ylabel("T")
        plt.legend()
        plt.grid()

    # Plot the errors
    plt.figure()
    plt.suptitle("Errors")
    plt.plot(P, error_CDS, label="CDS", marker="o")
    plt.plot(P, error_UDS, label="UDS", marker="o")
    plt.xlabel("Pe")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()

    plt.show()
    stop = True
