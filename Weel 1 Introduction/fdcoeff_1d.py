import numpy as np
import math
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
            M[i, j] = (x[i] - x0)**j / math.factorial(j)

    # Inversing M
    A = np.linalg.inv(M)
    
    # Scaling
    A = A / (dx0**np.arange(r))
    return A

def eval_fdcoeff_1d_general(fun: callable, x: List[float], a: int, b: int):
    """
    Computes coefficients of one-dimensional finite difference schemes on an even stencil.
    """
    u = fun(x)
    ux = np.zeros(len(x))
    for i in range(0, len(x)):
        a_i = max(0, i - a)
        b_i = min(len(x), i + b)
        x_i = x[a_i:b_i+1] # Stencil
        u_i = u[a_i:b_i+1]
        A = fdcoeff_1d_general(x_i, x[i])
        ux[i] = np.dot(A[1, :], u_i)
    return ux

def convergence_test_general():
    N = [15, 45, 135, 405]
    C = [0.0000000001, 0.5, 1.5, 2.5]

    fun = lambda x: 1.0/(1.0 + 25.0*x**2)
    diff_fun = lambda x: -50.0*x/(1.0 + 25.0*x**2)**2

    a = 2
    b = 2

    error = np.zeros((len(C), len(N)))
    roc = np.zeros(len(C))
    TL = np.zeros((len(C), len(N)))

    for i, c in enumerate(C):
        for j, n in enumerate(N):
            nn = int((n-1)/2 + 1)
            ds = 1/(nn-1)
            s = np.transpose(np.arange(0, nn))*ds
            xi = np.tanh(c * s)/np.tanh(c)
            x = np.zeros(2*nn-1)
            x[0 : nn] = -1 + xi
            x[nn-1 ::] = 1 - xi[nn-1::-1]
            u = fun(x)
            ux = diff_fun(x)
            ux_approx = eval_fdcoeff_1d_general(fun, x, a, b)

            x_test = np.linspace(-10, 10, 1000)

            plt.figure()
            #plt.plot(x, ux, label="Exact")
            plt.plot(x, ux_approx, label="Approx")
            plt.legend()

            plt.figure()
            plt.scatter(x, u)
            plt.plot(x_test, fun(x_test))
            plt.show()

            error[i, j] = np.abs(ux_approx - ux).max()

        _roc = np.polyfit(np.log(N), np.log(error[i, :]),1)
        roc[i] = _roc[0]
        TL[i, :] = np.polyval(_roc, np.log(N))
    
    # plot 2 subplots
    plt.figure()
    plt.suptitle(f"Convergence test with a = {a}, b = {b}")
    for i, c in enumerate(C):
        plt.loglog(N, error[i, :], label=f"C = {c}. Slope: {roc[i]:.2f}", marker="o")
        plt.semilogx(N, TL[i, :])
        plt.xlabel("N")
        plt.ylabel("Error")
        plt.legend()
    plt.show()
        
    stop = True


if __name__ == "__main__":
    

    if False:
        convergence_test_uniform()

    if True:
        convergence_test_general()

