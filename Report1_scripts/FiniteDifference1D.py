import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List
import scipy.sparse as sps
import sympy as sp

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

def diffmatrix_1d_uniform(der, N, dx, a: int, b: int):
    """
    Computes 1D Finite Difference differentiation matrix for derivative 'der' 
    on a grid with 'N' points and uniform grid spacing 'dx', using a stencil 
    'a' points to the left and 'b' points to the right on the interior of 
    the domain, and increasingly off-centered stencils with r=a+b+1 points
    near the domain boundaries.
    """
    r = a + b + 1
    D = sps.csr_matrix((N, N), dtype=float)

    A = fdcoeff_1d_uniform(a, b)
    for i in range(N):
        if i < a-1:
            x = np.arange(0, r)
        elif i > N - b - 1:
            x = np.arange(i - a, i + b + 1)
            x[-1] = x[-1] - N  # Periodic boundary condition
        else:
            x = np.arange(i - a, i + b + 1)
        D[i, x] = A[der, :] / dx**der
    return D

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
    for i in range(r):
        A[i, :] = A[i, :] / dx0**i

    return A

def FDmatrix(x: List[float], x0: float, extras: int = 6):
    """
    Computes coefficients of one-dimensional finite difference schemes on an even stencil.
    """
    r = len(x)
    dx0 = np.min(np.diff(x))
    #dx0 = 1
    x = x / dx0
    x0 = x0 / dx0
    M = np.zeros((r, r))
    M2 = np.zeros((r, r + extras))
    for i in range(r):
        for j in range(r):
            M[i, j] = (x[i] - x0)**j / math.factorial(j)

    for i in range(r):
        for j in range(r + extras):
            M2[i, j] = (x[i] - x0)**j / math.factorial(j)

    # # Inversing M
    # A = np.linalg.inv(M)
    # A2 =M2
    
    # # Scaling
    # for i in range(r):
    #     A[i, :] = A[i, :] / dx0**i
    #     A2[:, i] = A2[:, i] / dx0**i # NB MAYBE THIS IS WRONG

    return M, M2
    
def get_stencil_1d(x: List[float], x0: float, der: int):
    c = fdcoeff_1d_general(x, x0)
    cder = c[der, :]
    print("Coefficients for derivative", der)
    print([sp.nsimplify(n) for n in cder])
    print("The approximation is given by:")
    print(f"f_i^{der} = 1/dx^{der} * ({' + '.join([f'{sp.nsimplify(cder[i])}*f_(i+{xi+0.5})' for i, xi in enumerate(x)])})")
    return cder

def get_leading_truncation_error_1d(x: List[float], x0: float, der: int):
    extras = 2
    fE = np.zeros(len(x) + extras)
    fE[der] = 1
    m, m2 = FDmatrix(x, x0, extras)
    c = np.linalg.inv(m)
    fder = c[der, :]
    n = len(x)
    fFD = np.zeros(n + extras)
    for i in range(n):
        fFD += fder[i] * m2[i, :]
    
    e = fE - fFD

    e_index = np.nonzero(e)[0]
    # Get the leading term
    e_index = e_index[0]
    print("The leading truncation error term for derivative", der, "is:")
    print(f"e_i = {sp.nsimplify(e[e_index])} * dx^{int(e_index-der)} *f_i^{len(x)}")
    return e[e_index]

def get_east_minus_west_1d(x: List[float], x0: float, der: int):
    c = fdcoeff_1d_general(x, x0)
    fd = c[der, :].tolist()
    fde = [0] + fd
    fdw = fd + [0]
    fdew = np.array(fde) - np.array(fdw)
    print("The east minus west difference for derivative", der, "is:")
    print([sp.nsimplify(n) for n in fdew])
    print("The approximation is given by:")
    print(f"f_i^{der} = 1/dx^{der} * ({' + '.join([f'{sp.nsimplify(fdew[i])}*f_(i+{xi+0.5})' for i, xi in enumerate(x)])})")
    return fdew

    
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

def diffmatrix_1d_general(der,x,a,b):
    """
    Computes 1D Finite Difference differentiation matrix for derivative 'der' 
    on a grid with 'N' points and uniform grid spacing 'dx', using a stencil 
    'a' points to the left and 'b' points to the right on the interior of 
    the domain, and increasingly off-centered stencils with r=a+b+1 points
    near the domain boundaries.
    """
    N = len(x)
    D = sps.csr_matrix((N, N), dtype=float)
    r = a + b + 1
    
    for i in range(N):
        if i < a:
            a_i = 0
            b_i = r - 1
        elif i > N - b - 1:
            a_i = N - r
            b_i = N 
        else:
            a_i = i - a
            b_i = i + b
        x_i = x[a_i:b_i+1] # Stencil
        assert len(x_i) == r
        A_i = fdcoeff_1d_general(x_i, x[i])
        D[i, a_i:b_i+1] = A_i[der, :]

    Dtest = D.toarray().tolist()
    return D




if __name__ == "__main__":
    ders = [0, 2]
    x = [-1.5, -0.5, 0.5, 1.5]
    # x = [-2.5, -1.5, -0.5, 0.5]
    # x = [-1.5, - 0.5, 0.5]
    x0 = 0
    for der in ders:
        print("----- Stencil for derivative", der, "-----")
        get_stencil_1d(x, x0, der)
        print("----- Leading truncation error for derivative", der, "-----")
        get_leading_truncation_error_1d(x, x0, der)
        print("----- East minus West difference for derivative", der, "-----")
        get_east_minus_west_1d(x, x0, der)
        # New line
        print("\n \n")
    
    # raise Exception("This file is not meant to be executed on its own.")
    D_uniform = diffmatrix_1d_uniform(1, 5, 1, 1, 1)
    x = np.arange(0, 5)
    x0 = x[3]
    D_general = diffmatrix_1d_general(1, x, x0, 1, 1)
    stop = True
