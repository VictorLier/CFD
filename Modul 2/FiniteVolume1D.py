import numpy as np
import scipy.sparse as sps
import math
import time
import matplotlib.pyplot as plt

def generate_velocity_field(n, xf):
    '''
    Outputs UFw and UFe
    Insert velocity profile here
    '''
    UFw = np.ones(n)
    UFe = np.ones(n)

    return UFw, UFe


def convective_face_matrix(n, dx, Pe, UFw, UFe):
    Fe = Pe * UFe
    Fw = Pe * UFw

    return Fw, Fe



def coef_matrix(n,dx,Pe,fvscheme,Fw,Fe):
    D = -1/dx
    if fvscheme == 'cds':
        aw = (D-Fw/2)
        ae = (D+Fe/2)
        ap = -(ae+aw)+Fe-Fw
        if dx*Pe >= 2:
            print("Warning - Matrix a no longer diagonally dominant - Non physcial solutions!")
        return aw, ae ,ap
    
    if fvscheme == 'uds':
        Fw[Fw < 0] = 0
        aw = D - Fw
        Fe[Fe < 0] = 0
        ae = D + Fe
        ap = -(ae+aw)+Fe-Fw
        return aw, ae, ap
    
    raise NotImplementedError("fvscheme not implemented")


def impose_boundary(n, dx, problem, aW, aE, aP,BC):
    s = np.zeros(n)
    aW = aW[0:-1]
    aE = aE[0:-1]

    if problem == 1:
        Ta = BC[0,0]
        Tb = BC[0,1]
        aP[0] = aP[0]-aW[0]
        s[0] = s[0]-2*Ta*aW[0]
        aW[0] = 0
        aP[-1] = aP[-1] - aE[-1]
        s[-1] = s[-1] - 2*Tb*aE[-1]
        aE[-1] = 0
        return s, aW, aE, aP
    
    if problem == 2:
        Ta = BC[1,0]
        Db = BC[1,1]
        aP[0] = aP[0]-aW[0]
        s[0] = s[0]-2*Ta*aW[0]
        aW[0] = 0
        aP[-1] = aP[-1] + aE[-1]
        s[-1] = s[-1]-aE[-1]*dx*Db
        aE[-1] = 0
        return s, aW, aE, aP

    raise NotImplementedError("Problem not implemented")

def assemble_matrix(n,aW,aE,aP):

    data = np.array([aP, aW, aE])
    D = sps.diags(data,[0,-1,1])
    return D

def solve(A,s):
    st = time.perf_counter()
    T = sps.linalg.spsolve(A, s)
    et = time.perf_counter()

    solve_time = st - et

    return T, solve_time




if __name__=="__main__":
    n = 50
    L = 1
    dx = L/n
    P = 1
    Pe = 1
    problem = 1
    fvscheme = 'uds'

    BC = np.array([[0,1],[0,1]])

    if problem == 1:
        Texact = lambda x: (np.exp(Pe*x)-1)/(np.exp(Pe)-1)

    elif problem == 2:
        Texact = lambda x: 1/Pe*math.exp(-Pe)*(math.exp(Pe*x)-1)


    
    xf = np.arange(0,L+dx,dx)
    xc = np.arange (dx/2,L+dx/2,dx)

    UFw, UFe = generate_velocity_field(n, xf)

    Fw, Fe = convective_face_matrix(n,dx,Pe,UFw,UFe)

    aW, aE ,aP = coef_matrix(n, dx, Pe, fvscheme, Fw, Fe)

    s, aW, aE, aP = impose_boundary(n, dx, problem, aW, aE, aP, BC)

    D = assemble_matrix(n, aW, aE, aP)

    T, solve_time = solve(D,s)
    

    #fun = lambda x: (np.exp(Pe*x)-1)/(np.exp(Pe)-1)
    # Plot the results in a single figure
    x_plot = np.linspace(0, L, 1000)
    T_true_plot = Texact(x_plot)
    plt.figure()
    plt.suptitle("Test for P = {}".format(P))
    plt.plot(x_plot, T_true_plot, label="T_true")
    plt.plot(xc, T, label="T_CDS", marker="o")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.legend()
    plt.grid()
    plt.show()




    print("stop")
