import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List

from FiniteDifference1D import fdcoeff_1d_uniform, fdcoeff_1d_general, eval_fdcoeff_1d_general, diffmatrix_1d_uniform, diffmatrix_1d_general
from FiniteDifference2D import fdcoeff_2d_general

def convergence_test_1D_uniform():
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

def convergence_test_1D_general():
    """
    Exercise 222
    """
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
            D = diffmatrix_1d_general(1, x, a, b)
            ux_approx = D @ u
            # ux_approx = eval_fdcoeff_1d_general(fun, x, a, b)

            error[i, j] = np.abs(ux_approx - ux).max()
            # Get relative error
            # _e = np.abs(ux_approx - ux)/np.abs(ux)
            # error[i, j] = _e.max()

        _roc = np.polyfit(np.log(N), np.log(error[i, :]), 1)
        roc[i] = _roc[0]
        TL[i, :] = np.polyval(_roc, np.log(N))

        for j, n in enumerate(N):
            print(N[j], error[i, j])
    
    # plot 2 subplots
    plt.figure()
    plt.suptitle(f"Convergence test with a = {a}, b = {b}")
    for i, c in enumerate(C):
        plt.loglog(N, error[i, :], label=f"C = {c}. Slope: {roc[i]:.2f}", marker="o")
        #plt.scatter(np.log(N), TL[i, :])
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
        
    stop = True

def convergence_test_2D():
    """
    Exercise 223
    """
    #Input
    N = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
    L = 1
    a = 1
    b = 1
    DERx = 1
    DERy = 1

    #Initialize
    k = 2*math.pi/L

    D = L/(N-1)
    error = np.zeros(len(N))

    for i, n in enumerate(N):
        x = np.linspace(0, L, n)
        y = np.linspace(0, L, n)

        X, Y = np.meshgrid(x,y)

        fun = np.sin(k*X)*np.sin(k*Y)
        dfun = k*k*np.cos(k*X)*np.cos(k*Y)

        D_xy = fdcoeff_2d_general(DERx, DERy, x, y, a, b, n)

        fun_flat = fun.flatten()
        dfun_flat = D_xy.dot(fun_flat)

        dfun_aprrox = dfun_flat.reshape(fun.shape)

        error[i] = np.abs(dfun_aprrox - dfun).max()

    _roc = np.polyfit(np.log(N), np.log(error), 1)
    roc = _roc[0]
    TL = np.polyval(_roc, np.log(N))

    # plot 2 subplots
    plt.figure()
    plt.suptitle("Convergence test")
    plt.loglog(N, error, label=f"Slope: {roc:.2f}", marker="o")
    # plt.semilogx(N, TL)
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()

def poiseuille_flow():
    """
    Exercise 241
    """
    N = np.round(np.logspace(np.log10(3), np.log10(1000), 25)).astype(int)

    ERROR = np.zeros(len(N))


    fun = lambda z: 4*z*(1-z)

    for i, n in enumerate(N):
        #A matrix
        A = np.zeros((n,n))
        np.fill_diagonal(A,-2)
        np.fill_diagonal(A[:,1:],1)
        np.fill_diagonal(A[1:],1)
        A[0,0] = 1
        A[0,1] = 0
        A[n-1,n-1] = 1
        A[n-1,n-2] = 0
        
        # B vector
        b_vec = np.full(n,8)
        b_vec[0] = 0
        b_vec[n-1] = 0
        b = - 1/n**2 * b_vec
        
        #Solve u
        u = np.linalg.solve(A,b)

        z = np.linspace(0,1,n)
        u_exact = fun(z)

        ERROR[i] = np.max(abs(u-u_exact))

    data = np.vstack((N,ERROR))      

if __name__ == "__main__":

    convergence_test_1D_uniform()
    convergence_test_1D_general()
    convergence_test_2D()
    poiseuille_flow()

    plt.show()