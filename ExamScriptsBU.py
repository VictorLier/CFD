# The main functions in this scripts were developed in the context of the reports/projects.
# Therefore, Victor Lier Hansen (s204389) might use equavalent functions in the exam.
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List
import scipy.sparse as sps
import sympy as sp
import numpy as np
from typing import List
import numpy as np
from scipy.linalg import toeplitz
import numpy as np


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

def get_J_backup(x: List[float], x0: float, der: int):
    # Backup see practice exam 2
    N = 10
    dx = 1 / N
    x = [i * dx + dx / 2 for i in range(N)]
    f = get_east_minus_west_1d(x, x0, der)
    c = np.zeros(N)
    c[0] = 3
    c[1] = -7
    c[2] = 1
    c[N - 1] = 3
    r = np.zeros(N)
    r[0] = 3
    r[1] = 3
    r[N - 2] = 1
    r[N - 1] = -7
    J = -1 / (8 * dx) * toeplitz(c, r)
    return J

def get_stability_from_matrix(x, x0, der, N, dt, periodic = True, plot = True):
    J = np.zeros((N, N))
    dx = 1 / N

    _x = np.array([x[0]-1] + x)+0.5
    a = int(np.abs(np.min(_x-x0))) # Nb maybe be a bit careful with this
    b = int(np.abs(np.max(_x-x0))) # Nb maybe be a bit careful with this
    r = len(x) + 1

    #################### CHANGE THIS ACCORDING TO PROBLEM #####################
    # Modify the c vector to the correct values e.g. a-variables can be imposed
    c = get_east_minus_west_1d(x, x0, der, _print=False)
    c = c.copy()*(-1/dx)
    ###########################################################################
    if periodic:
        for i in range(N):
            if i < a:
                p = np.arange(i - a, i + b + 1) # Periodic boundary condition
            elif i > N - b - 1:
                p = np.arange(i - a, i + b + 1)
                p[-1] = p[-1] - N  # Periodic boundary condition
            else:
                p = np.arange(i - a, i + b + 1)
            J[i, p] = c / dx**der

    else:
        for i in range(N):
            if i < a:
                p = np.arange(0, i + b + 1)
                J[i, p] = c[-len(p):] / dx**der
            elif i > N - b - 1:
                p = np.arange(i - a, N)
                J[i, p] = c[:len(p)] / dx**der
            else:
                p = np.arange(i - a, i + b + 1)
                J[i, p] = c / dx**der
   
    eig = np.linalg.eigvals(J) * dt
    # Check for stability
    if 1 + np.max(np.abs(eig)) > 1:
        print(f"Unstable")
    else:
        print(f"Stable")

    if plot:
        # Show position of values using spy
        plt.figure()
        # plt.title(f"Jacobian for derivative {der}")
        plt.spy(J)
        # Change the ticks to start from 1
        plt.xticks(np.arange(0, N, 1), np.arange(1, N+1, 1))
        plt.yticks(np.arange(0, N, 1), np.arange(1, N+1, 1))
        # Add minor ticks. They should be at the half points
        plt.gca().set_xticks(np.arange(-.5, N, 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, N, 1), minor=True)
        # add grid for the minor ticks
        plt.grid(which='minor', color='k', linestyle='-')
        # Shown the values for each element using sp.nsimplify
        if False:
            for i in range(N):
                for j in range(N):
                    plt.text(j, i, sp.nsimplify(J[i, j]), ha='center', va='center', color='r')


        # Show eigenvalues
        plt.figure()
        # plt.title(f"Eigenvalues of Jacobian for Cr = {Cr}")
        plt.axvline(x=0, color='r', linestyle='--', label="Midtpoint rule limit")
        plt.scatter(eig.real, eig.imag, label="Eigenvalues")
        plt.xlabel("Re ($\lambda \Delta t$)")
        plt.ylabel("Im $\lambda \Delta t$")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid()
    

    return J

def get_eigenvalues_plot():
    n = np.linspace(2, 100, 1000)
    th = 2 * np.pi / n

    #################### CHANGE THIS ACCORDING TO PROBLEM #####################
    alpha = 1/3
    lam = 2 * alpha * (np.cos(th) - 1)
    _label = f'$ α ={alpha}$ '
    ###########################################################################

    plt.figure()
    # Use emtpy circles
    plt.plot(np.real(lam), np.imag(lam), 'o-', fillstyle='none', label = _label)
    plt.axvline(x=0, color='r', linestyle='--', label="Midtpoint rule limit")
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('Re($\lambda \Delta t$)')
    plt.ylabel('Im($\lambda \Delta t$)')
    plt.legend() 

if __name__ == "__main__":
    ders = [1]
    # x = [-1.5, -0.5, 0.5, 1.5] # Exam 1
    # x = [-1.5, -0.5, 0.5] # Exam 2 and Exam 2019
    x = [-0.5, 0.5]
    
    x0 = 0
    periodic = False
    M, M2 = FDmatrix(x, x0, _print=True)

    for der in ders:
        print("\n")
        print("-------------- Stencil for derivative", der, " --------------")
        get_stencil_1d(x, x0, der)
        print("------ Leading truncation error for derivative", der, "------")
        get_leading_truncation_error_1d(x, x0, der)
        print("----- East minus West difference for derivative", der, "-----")
        get_east_minus_west_1d(x, x0, der)
        if False:
            print("---------------- Stability analysis", der, " ----------------")
            N = 10
            Cr = 0.5
            dt = Cr * 1 / N
            get_stability_from_matrix(x, x0, der, N, dt, periodic)
        print("\n")
    
    get_eigenvalues_plot()
    stop = True
    plt.show()    
    