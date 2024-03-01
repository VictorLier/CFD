import numpy as np
import scipy as sps
import matplotlib.pyplot as plt


def convectiveVelocityField(problem, n, xf):
    '''
    Generates the face velocities
    (UFw, UFe, VFs,VFn)
    '''

    if problem == 1:
        UFw = np.ones(n+1)
        UFe = np.ones(n+1)
        VFs = np.zeros(n+1)
        VFn = np.zeros(n+1)
        return UFw, UFe, VFs,VFn

    if problem == 2:
        UFw = np.ones(n+1)
        UFe = np.ones(n+1)
        VFs = np.ones(n+1)
        VFn = np.ones(n+1)
        return UFw, UFe, VFs,VFn



if __name__=="__main__":
    n = 10
    L = 1
    Pe = 10
    problem = 1
    fvscheme = 'cds'

    dx = L/n
    xf = np.arange(0,L+dx,dx)
    xc = np.arange(dx/2,L-dx/2+dx,dx)


    print("Stop")