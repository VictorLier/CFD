# The main functions in this scripts were developed in the context of the reports/projects.
# Therefore, Victor Lier Hansen (s204389) might use equavalent functions in the exam.
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List
import sympy as sp
from typing import List


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

def FDmatrix(x: List[float], x0: float, extras: int = 2, _print = False):
    """
    Computes coefficients of one-dimensional finite difference schemes on an even stencil.
    """
    r = len(x)
    # dx0 = np.min(np.diff(x))
    # dx0 = 1
    # x = x / dx0
    # x0 = x0 / dx0
    M = np.zeros((r, r))
    M2 = np.zeros((r, r + extras))
    for i in range(r):
        for j in range(r):
            M[i, j] = (x[i] - x0)**j / math.factorial(j)

    for i in range(r):
        for j in range(r + extras):
            M2[i, j] = (x[i] - x0)**j / math.factorial(j)

    if _print:
        # Show the matrix M using sp.nsimplify
        print("Matrix M:")
        print([[sp.nsimplify(n) for n in m] for m in M])
        M_inv = np.linalg.inv(M)
        print("Matrix M_inv:")
        print([[sp.nsimplify(n) for n in m] for m in M_inv])
        print("Matrix M2:")
        print([[sp.nsimplify(n) for n in m] for m in M2])

    return M, M2
    
def get_stencil_1d(x: List[float], x0: float, der: int):
    c = fdcoeff_1d_general(x, x0)
    cder = c[der, :]
    cder = [0 if np.abs(_cder) <= 1e-10 else _cder for _cder in cder]
    print("Coefficients for derivative", der)
    print([sp.nsimplify(n) for n in cder])
    print("The approximation for the eastern boundary is given by:")
    print(f"f^{der} = 1/dx^{der} * ({' + '.join([f'{sp.nsimplify(cder[i])}*f_(i+{xi+0.5})' for i, xi in enumerate(x)])})")
    if True:
        print("The approximation for the western boundary is given by:")
        print(f"f^{der} = 1/dx^{der} * ({' + '.join([f'{sp.nsimplify(cder[i])}*f_(i+{xi-0.5})' for i, xi in enumerate(x)])})")
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

    e = [0 if np.abs(_e) <= 1e-10 else _e for _e in e]
    e_index = np.nonzero(e)[0]

    e_index = e_index[0]

    # Get the leading term
    # e_index = e_index[0]
    print("The error vector is:")
    print([sp.nsimplify(n) for n in e])
    print("The leading truncation error term for derivative", der, "is:")
    print(f"e = {sp.nsimplify(e[e_index])} * dx^{int(e_index-der)} *f_i^{len(x)}")
    return e[e_index]

def get_east_minus_west_1d(x: List[float], x0: float, der: int, _print = True):
    c = fdcoeff_1d_general(x, x0)
    fd = c[der, :].tolist()
    fde = [0] + fd
    fdw = fd + [0]
    x_ew = x.copy()
    x_ew.append(x[-1] + 1)
    fdew = np.array(fde) - np.array(fdw)
    fdew = np.array([0 if np.abs(_fdew) <= 1e-10 else _fdew for _fdew in fdew])
    if _print:
        print("The east minus west difference for derivative", der, "is:")
        print([sp.nsimplify(n) for n in fdew])
        print("The approximation is given by:")
        print(f"f_ew^{der} = 1/dx^{der} * ({' + '.join([f'{sp.nsimplify(fdew[i])}*f_(i+{xi-0.5})' for i, xi in enumerate(x_ew)])})")
    return fdew


def get_eigenvalues_plot():
    n = np.linspace(2, 100, 1000)
    th = 2 * np.pi / n
    v = np.linspace(0, 2*np.pi, 1000)

    alpha = 1/3
    # alpha = 13/25
    lam = alpha/12*(-2*np.cos(2*th)+32*np.cos(th)-30)
    _label = f'$ α ={sp.nsimplify(alpha)}$ '

    plt.figure()
    # Use emtpy circles
    plt.plot(np.real(lam), np.imag(lam), 'o-', fillstyle='none', label = _label)
    
    # plt.plot(np.cos(v)-1, np.sin(v), label = "Forward Euler") # Used for Q15a

    # plt.axvline(x=-2.785, color='r', linestyle='--', label="x-crossing") #Used for Q15b
    # plt.axhline(y=2.828, color='b', linestyle='--', label="y-crossing") #Used for Q15b
    # plt.axhline(y=-2.828, color='b', linestyle='--') #Used for Q15b

    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('Re($\lambda \Delta t$)')
    plt.ylabel('Im($\lambda \Delta t$)')
    plt.xlim(-3,0.5)
    plt.ylim(-3,3)
    plt.legend() 

if __name__ == "__main__":
    der = 1
    x = [-0.5, 0.5]
    x0 = 0

    M, M2 = FDmatrix(x, x0, _print=True)

    print("\n")
    print("-------------- Stencil for derivative", der, " --------------")
    get_stencil_1d(x, x0, der)
    print("------ Leading truncation error for derivative", der, "------")
    get_leading_truncation_error_1d(x, x0, der)
    print("----- East minus West difference for derivative", der, "-----")
    get_east_minus_west_1d(x, x0, der)
    
    get_eigenvalues_plot()
    stop = True
    plt.show()    
    