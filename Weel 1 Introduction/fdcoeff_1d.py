import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List
from scipy.sparse import lil_matrix, csr_matrix 

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
    D = lil_matrix((N, N), dtype=float)

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
    D = csr_matrix((N, N), dtype=float)
    
    for i in range(N):
        a_i = max(0, i - a)
        b_i = min(len(x), i + b)
        x_i = x[a_i:b_i+1] # Stencil
        A_i = fdcoeff_1d_general(x_i, x[i])
        D[i, a_i:b_i+1] = A_i[der, :]
    return D




if __name__ == "__main__":
    # raise Exception("This file is not meant to be executed on its own.")
    D_uniform = diffmatrix_1d_uniform(1, 5, 1, 1, 1)
    x = np.arange(0, 5)
    x0 = x[3]
    D_general = diffmatrix_1d_general(1, x, x0, 1, 1)
    stop = True
