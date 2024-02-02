import numpy as np
import matplotlib.pyplot as plt
from typing import List

def fdcoeff_1d_uniform(a: int, b: int):
    """
    Computes coefficients of one-dimensional finite difference schemes on an even stencil.
    """

    r = a + b + 1
    M = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            M[i, j] = (i - a)**j / np.math.factorial(j)

    # Inversing M
    A = np.linalg.inv(M)
    return A

def convergence_test_uniform():
    """
    Exercise 221
    Test the coefficients of one-dimensional finite difference schemes on an even stencil.
    """

    stencil = [(0, 1), (1, 1), (2, 2)]

    N = np.array([5, 10, 20, 40, 80, 160, 320])

    L = 10
    k = 2*np.pi/L
    fun = lambda x: np.sin(k*x)
    diff_fun = lambda x: k*np.cos(k*x)
    

    error = np.zeros((len(stencil), len(N)))
    roc = np.zeros(len(stencil))
    TL = np.zeros((len(stencil), len(N)))

    for i, (a, b) in enumerate(stencil):
        A = fdcoeff_1d_uniform(a, b)
        for j, n in enumerate(N):
            x = np.linspace(0, L, n)
            dx = L/(n-1)
            u = fun(x)
            ux = diff_fun(x)
            ux_approx = np.array([np.dot(A[1,:], u[i-a:i+b+1]) for i in range(a, n-b)])/dx
            error[i, j] = np.abs(ux_approx - ux[a:n-b]).max()
        

        roc[i] = np.polyfit(np.log(N), np.log(error[i, :]), 1)[0]
        TL[i, :] = np.polyval(np.polyfit(np.log(N), np.log(error[i, :]), 1), np.log(N))
    
    # plot 2 subplots
    plt.figure()
    plt.suptitle("Convergence test")
    for i, (a, b) in enumerate(stencil):
        plt.loglog(N, error[i, :], label=f"Stencil: a = {a}, b = {b}. Slope: {roc[i]:.2f}", marker="o")
        plt.semilogx(N, TL[i, :])
        plt.xlabel("N")
        plt.ylabel("Error")
        plt.legend()
    plt.show()
        
    stop = True


def fdcoeff_1d_general(x: List[float], x0: float):
    """
    Computes coefficients of one-dimensional finite difference schemes on an even stencil.
    """

    r = len(x)
    dx0 = np.min(np.diff(x))
    #dx0 = 1
    x = x / dx0
    x0 = x0 / dx0
    M = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            M[i, j] = (x[i] - x0)**j / np.math.factorial(j)

    # Inversing M
    A = np.linalg.inv(M)
    
    # Scaling
    A = A / dx0**np.arange(r)
    return A


if __name__ == "__main__":
    

    if False:
        convergence_test_uniform()

    if True:
        x = np.array([0, 1, 2])
        x0 = 1
        A = fdcoeff_1d_general(x, x0)

