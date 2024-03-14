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
    ''''
    This function calculates the convective face matrix with
    '''
    Fe = Pe * UFe * dx
    Fw = Pe * UFw * dx
    Fs = Pe * VFs * dx
    Fn = Pe * VFn * dx

    return Fw, Fe, Fs, Fn

def coef_matrix(n,dx,Pe,fvscheme,Fw,Fe,Fs,Fn):
    D = -dx/dx # Nb should be made individual for w,e,s,n if dx,dy are different
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
        ## North
        aP[-1,:] = aP[-1,:] + aN[-1,:]
        s[-1,:] = s[-1,:] - aN[-1,:]*dx*DTgn
        aN[-1,:] = 0

        ## South
        aP[0,:] = aP[0,:] + aS[0,:]
        s[0,:] = s[0,:] + aS[0,:]*dx*DTgs
        aS[0,:] = 0
        
        
        # Dirichlet boundary conditions
        ## West
        aP[:,0] = aP[:,0] - aW[:,0]
        s[:,0] = s[:,0] - 2*Tgw*aW[:,0]
        aW[:,0] = 0
        ## East
        aP[:,-1] = aP[:,-1] - aE[:,-1]
        s[:,-1] = s[:,-1] - 2*Tge*aE[:,-1]
        aE[:,-1] = 0
        return s, aW, aE, aS, aN, aP
    
    if problem == 2:
        DTgn = 0
        Tgs = xc
        Tgw = 0
        Tge = 1
        # Nuemann boundary conditions
        ## North
        aP[-1,:] = aP[-1,:] + aN[-1,:]
        s[-1,:] = s[-1,:] - aN[-1,:]*dx*DTgn
        aN[-1,:] = 0
        
        # Dirichlet boundary conditions
        ## West
        aP[:,0] = aP[:,0] - aW[:,0]
        s[:,0] = s[:,0] - 2*Tgw*aW[:,0]
        aW[:,0] = 0
        ## East
        aP[:,-1] = aP[:,-1] - aE[:,-1]
        s[:,-1] = s[:,-1] - 2*Tge*aE[:,-1]
        aE[:,-1] = 0
        ## South
        aP[0,:] = aP[0,:] - aS[0,:]
        s[0,:] = s[0,:] - 2*Tgs*aS[0,:]
        aS[0,:] = 0
        
        return s, aW, aE, aS, aN, aP
    
    raise NotImplementedError("problem not implemented")

def assemble_matrix(n, aW, aE, aS, aN, aP):
    """
    return D
    """
    data = np.array([aP.flatten('F'), aE.flatten('F'), aN.flatten('F'), aS.flatten('F'), aW.flatten('F')])
    D = sps.spdiags(data, [0, -n, -1, 1, n], n*n, n*n, format = 'csr').T
    return D

def solve(A, s, n):
    st = time.perf_counter()
    T = sps.linalg.spsolve(A, s.flatten('F'))
    et = time.perf_counter()

    solve_time = et - st

    return T.reshape(n, n, order='F'), solve_time

def extrapolate_temperature_field_to_walls(n, dx, fvscheme, problem, T, xc):
    TT = np.zeros((n+2,n+2))
    TT[1:-1,1:-1] = T
    if problem == 1:
        if fvscheme == 'cds':    
            TT[:,0] = 0
            TT[:,-1] = 1
            TT[0,:] = TT[1,:] - 1/2 * dx * 0 # NB 0 is bundary condition (D_a)
            TT[-1,:] = TT[-2,:] + 1/2 * dx * 0 # NB 0 is bundary condition (D_b)
            return TT
    
        if fvscheme == 'uds':
            TT[:,0] = 2*0 - TT[:,1]
            TT[:,-1] = TT[:,-2]
            TT[0,:] = TT[1,:] - dx * 0 # NB 0 is bundary condition (D_a)
            TT[-1,:] = TT[-2,:]
            return TT
        
    if problem == 2:
        if fvscheme == 'cds':
            _xc = np.zeros(len(xc)+2)
            _xc[1:-1] = xc
            _xc[0] = xc[0] - dx/2
            _xc[-1] = xc[-1] + dx/2
            TT[:,0] = 0
            TT[:,-1] = 1
            TT[0,:] = _xc
            TT[-1,:] = TT[-2,:]
            return TT

        if fvscheme == 'uds':
            _xc = np.zeros(len(xc)+2)
            _xc[1:-1] = xc
            _xc[0] = xc[0] - dx/2
            _xc[-1] = xc[-1] + dx/2
            TT[:,0] = TT[:,1]
            TT[:,-1] = 2*1 - TT[:,-2] #NB 1 is bundary condition
            TT[0,:] = 2*_xc - TT[1,:]
            TT[-1,:] = TT[-2,:]
            return TT

def extrapolate_gradients_to_walls(n, dx, fvscheme, problem, TT, xc):
    dT = np.zeros((n + 2,n + 2))
    if problem == 1:

        dT[:,0] = 2 / dx * (TT[:,1] - 0)
        dT[:,-1] = 2 / dx * (1 - TT[:,-2])
        dT[0,:] =  0 # NB 0 is bundary condition (D_a)
        dT[-1,:] =  0 # NB 0 is bundary condition (D_b)
        return dT

    if problem == 2:
        _xc = np.zeros(len(xc)+2)
        _xc[1:-1] = xc
        _xc[0] = xc[0] - dx/2
        _xc[-1] = xc[-1] + dx/2

        dT[:,0] = 2 / dx * (TT[:,1] - 0)
        dT[:,-1] = 2 / dx * (1 - TT[:,-2])
        dT[0,:] = 2 / dx * (TT[1,:] - _xc)
        dT[-1,:] = 0
        return dT


def plot_temperature_field(xc, TT, dx, L):
    _xc = np.zeros(len(xc)+2)
    _xc[1:-1] = xc
    _xc[0] = xc[0] - dx/2
    _xc[-1] = xc[-1] + dx/2
    plt.figure()
    # Make a grid plot with temperature as the color usin pcolormesh
    plt.pcolormesh(_xc, _xc, TT)
    # Add label to colorbar
    plt.colorbar(label='Temperature')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.title('Temperature distribution')

    plt.figure()
    plt.contourf(_xc, _xc, TT)
    plt.colorbar(label='Temperature')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.title('Temperature distribution')

    # Plot the surface 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(_xc, _xc)
    ax.plot_surface(X, Y, TT, cmap='viridis')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('Temperature')
    ax.set_title('Temperature distribution')


    # fig, ax = plt.subplots(dpi = 200)
    # plt.imshow(T, interpolation='none')
    # minor_locator = AutoMinorLocator(2)
    # ax.yaxis.set_minor_locator(minor_locator)
    # ax.xaxis.set_minor_locator(minor_locator)
    # plt.xticks(ticks = xc, labels = 'x-axis', rotation = 'vertical')
    # plt.yticks(ticks = xc, labels = 'y-axis')
    # ax.grid(True, which='minor')

def GlobalConservation(TT, dT, dx, Pe, xc, UFw, UFe, VFs, VFn, problem, fvscheme):
    """
    return Conservation error
    """
    PeuTw = Pe * UFw[:,0] * TT[1:-1,0]
    PeuTe = Pe * UFe[:,-1] * TT[1:-1,-1]
    PeuTs = Pe * VFs[0,:] * TT[0,1:-1]
    PeuTn = Pe * VFn[-1,:] * TT[-1,1:-1]
    xflux = np.sum((PeuTe-dT[1:-1,-1]) - (PeuTw-dT[1:-1,0])) * dx
    yflux = np.sum((PeuTn-dT[-1,1:-1]) - (PeuTs-dT[0,1:-1])) * dx
    Flux = xflux + yflux
    return Flux


def do_simulation(n, L, Pe, problem, fvscheme, plot = False):
    dx = L/n
    xf = np.arange(0,L+dx,dx)
    xc = np.arange(dx/2,L-dx/2+dx,dx)

    UFw, UFe, VFs,VFn = convectiveVelocityField(problem, n, xf)
    Fw, Fe, Fs, Fn = convective_face_matrix(n, dx, Pe, UFw, UFe, VFs, VFn)
    aW, aE, aS, aN, aP = coef_matrix(n,dx,Pe,fvscheme,Fw,Fe,Fs,Fn)
    s, aW, aE, aS, aN, aP = impose_boundary(n, dx, xc, problem, aW, aE, aS, aN, aP)
    D = assemble_matrix(n, aW, aE, aS, aN, aP)
    T, solve_time = solve(D,s,n)
    TT = extrapolate_temperature_field_to_walls(n, dx, fvscheme, problem, T, xc)
    dT = extrapolate_gradients_to_walls(n, dx, fvscheme, problem, TT, xc)
    Flux = GlobalConservation(TT, dT, dx, Pe, xc, UFw, UFe, VFs, VFn, problem, fvscheme)
    
    if plot:
        plot_temperature_field(xc, TT, dx, L)
    return T, TT, dT, Flux, solve_time



if __name__=="__main__":
    n = 46
    L = 1
    # P = 1.5
    # Pe = P * n
    Pe = 7
    problem = 2
    fvscheme = 'cds'

    T, TT, dT, Flux, solve_time = do_simulation(n, L, Pe, problem, fvscheme, plot = True)


    plt.show()
