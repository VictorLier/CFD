import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import AutoMinorLocator


def convectiveVelocityField(problem, n, xf):
    '''
    Generates the face velocities
    (UFw, UFe, VFs,VFn)
    '''

    if problem == 1:
        # Matrix of ones
        UFw = np.ones((n,n))
        UFe = np.ones((n,n))
        VFs = np.zeros((n,n))
        VFn = np.zeros((n,n))
        return UFw, UFe, VFs,VFn

    if problem == 2:
        UFw = -np.tile(xf[:n],(n,1))
        UFe = -np.tile(xf[1:],(n,1))
        VFs = np.tile((xf[:n]).reshape(n,1),(1,n))
        VFn = np.tile((xf[1:]).reshape(n,1),(1,n))
        return UFw, UFe, VFs,VFn
    
def convective_face_matrix(n, dx, Pe, UFw, UFe, VFs, VFn):
    Fe = Pe * UFe
    Fw = Pe * UFw
    Fs = Pe * VFs
    Fn = Pe * VFn

    return Fw, Fe, Fs, Fn

def coef_matrix(n,dx,Pe,fvscheme,Fw,Fe,Fs,Fn):
    D = -1/dx
    if fvscheme == 'cds':
        aW = (D-Fw/2)
        aE = (D+Fe/2)
        aS = (D-Fs/2)
        aN = (D+Fn/2)
        aP = -(aE+aW+aN+aS)+Fe-Fw+Fn-Fs
        if dx*Pe >= 2:
            print("Warning - Matrix a no longer diagonally dominant - Non physcial solutions!")
        return aW, aE ,aS, aN, aP
    
    if fvscheme == 'uds':        
        aE = D + np.minimum(0,Fe)
        aW = D + np.minimum(0,-Fw)
        aN = D + np.minimum(0,Fn)
        aS = D + np.minimum(0,-Fs)
        aP = -(aE+aW+aN+aS)+Fe-Fw+Fn-Fs
        return aW, aE, aS, aN, aP
    
    raise NotImplementedError("fvscheme not implemented")

def impose_boundary(n, dx, xc, problem, aW, aE, aS, aN, aP):
    """
    return s, aW, aE, aS, aN, aP
    """
    s = np.zeros((n, n))
    
    if problem == 1:
        DTgn = 0
        DTgs = 0
        Tgw = 0
        Tge = 1
        # Nuemann boundary conditions
        aP[0,:] = aP[0,:] + aN[0,:]
        aP[-1,:] = aP[-1,:] + aS[-1,:]
        s[0,:] = s[0,:] - aN[0,:]*dx*DTgn
        s[-1,:] = s[-1,:] + aS[-1,:]*dx*DTgs
        aS[0,:] = 0
        aN[-1,:] = 0
        
        # Dirichlet boundary conditions
        aP[:,0] = aP[:,0] - aW[:,0]
        aP[:,-1] = aP[:,-1] - aE[:,-1]
        s[:,0] = s[:,0] - 2*Tgw*aW[:,0]
        s[:,-1] = s[:,-1] - 2*Tge*aE[:,-1]
        aW[:,0] = 0
        aE[:,-1] = 0
        return s, aW, aE, aS, aN, aP
    
    if problem == 2:
        DTgn = 0
        DTgs = xc
        Tgw = 0
        Tge = 2
        # Nuemann boundary conditions
        aP[0,:] = aP[0,:] + aN[0,:]
        aP[-1,:] = aP[-1,:] + aS[-1,:]
        s[0,:] = s[0,:] - aN[0,:]*dx*DTgn
        s[-1,:] = s[-1,:] + aS[-1,:]*dx*DTgs
        
        # Dirichlet boundary conditions
        aP[:,0] = aP[:,0] - aW[:,0]
        aP[:,-1] = aP[:,-1] - aE[:,-1]
        s[:,0] = s[:,0] - 2*Tgw*aW[:,0]
        s[:,-1] = s[:,-1] - 2*Tge*aE[:,-1]
        return s, aW, aE, aS, aN, aP
    
    raise NotImplementedError("problem not implemented")

def assemble_matrix(n, aW, aE, aS, aN, aP):
    """
    return D
    """
    data = np.array([aP.flatten('F'), aE.flatten('F'), aN.flatten('F'), aS.flatten('F'), aW.flatten('F')])
    D = sps.spdiags(data, [0, -n, -1, 1, n], n*n, n*n, format = 'csr').T
    return D

def solve(A, s):
    st = time.perf_counter()
    T = sps.linalg.spsolve(A, s.flatten('F'))
    et = time.perf_counter()

    solve_time = et - st

    return T.reshape(n,n,order='F'), solve_time

def extrapolate_temperature_field_to_walls(n, dx, fvscheme, problem, T):
    TT = np.zeros((n+2,n+2))
    TT[1:-1,1:-1] = T
    stop = True

def surface_plot(xc, T):
    fig = plt.figure()
    # Make a grid plot with temperature as the color usin pcolormesh
    plt.pcolormesh(xc, xc, np.flipud(T))
    # Add label to colorbar
    plt.colorbar(label='Temperature')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.title('Temperature distribution')


    # fig, ax = plt.subplots(dpi = 200)
    # plt.imshow(T, interpolation='none')
    # minor_locator = AutoMinorLocator(2)
    # ax.yaxis.set_minor_locator(minor_locator)
    # ax.xaxis.set_minor_locator(minor_locator)
    # plt.xticks(ticks = xc, labels = 'x-axis', rotation = 'vertical')
    # plt.yticks(ticks = xc, labels = 'y-axis')
    # ax.grid(True, which='minor')

if __name__=="__main__":
    n = 10
    L = 1
    P = 1
    Pe = P * n
    problem = 2
    fvscheme = 'uds'

    dx = L/n
    xf = np.arange(0,L+dx,dx)
    xc = np.arange(dx/2,L-dx/2+dx,dx)

    UFw, UFe, VFs,VFn = convectiveVelocityField(problem, n, xf)
    Fw, Fe, Fs, Fn = convective_face_matrix(n, dx, Pe, UFw, UFe, VFs, VFn)
    aW, aE, aS, aN, aP = coef_matrix(n,dx,Pe,fvscheme,Fw,Fe,Fs,Fn)
    s, aW, aE, aS, aN, aP = impose_boundary(n, dx, xc, problem, aW, aE, aS, aN, aP)
    D = assemble_matrix(n, aW, aE, aS, aN, aP)
    T, solve_time = solve(D,s)
    extrapolate_temperature_field_to_walls(n, dx, fvscheme, problem, T)
    surface_plot(xc, T)


    plt.show()
    print("Stop")