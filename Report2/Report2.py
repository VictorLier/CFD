import numpy as np
import scipy as sp

class CFDSim:
    def __init__(self, n, Re) -> None:
        self.n = n
        self.Re = Re
        self.StaggeredMesh2d()
        self.NS2dHfunctions()

    def StaggeredMesh2d(self):
        self.dx = 1/self.n   # cell size in x,y
        self.xf = np.arange(0, 1 + self.dx, self.dx)   # cell face coordinate vector, 1D
        self.xc = np.arange(self.dx/2, 1 - self.dx/2, self.dx)  # cell center coordinate vector, 1D
        self.xb = np.zeros(n+2)
        self.xb[0] = 0
        self.xb[-1] = 1
        self.xb[1:-1] = self.xc     # cell center coordinate vector incl boundaries
        self.Xu, self.Yu = np.meshgrid(self.xf, self.xb)    # u-grid coordinate array
        self.Xv, self.Yv = np.meshgrid(self.xb, self.xf)    # v-grid coordinate array
        self.Xp, self.Yp = np.meshgrid(self.xc, self.xc)    # p-grid coordinate array
        self.Xi, self.Yi = np.meshgrid(self.xf, self.xf)    # fd-grid coordinate array
        self.u = np.zeros((self.n+2, self.n+1))
        self.u[1:-1,:] = self.Xu[1:-1,:] * (1 - self.Xu[1:-1,:])
        self.u[-1,:] = 1
        self.v = np.zeros((self.n+1, self.n+2))
        self.v[:,1:-1] = self.Yv[:,1:-1] * (1 - self.Yv[:,1:-1])
        self.v[:,-1] = 1



    def NS2dHfunctions(self):
        # East and west - slide 14
        # ue = np.zeros((self.n, self.n-1))
        ue = (self.u[1:self.n+1, 2:self.n+1] + self.u[1:self.n+1, 1:self.n])/2
        # uw = np.zeros((self.n, self.n-1))
        uw = (self.u[1:self.n+1, 1:self.n] + self.u[1:self.n+1, 0:self.n-1])/2
        uxe = np.zeros((self.n, self.n-1))
        uxe = (self.u[1:self.n+1, 2:self.n+1] - self.u[1:self.n+1, 1:self.n])/self.dx
        uxw = np.zeros((self.n, self.n-1))
        uxw = (self.u[1:self.n+1, 1:self.n] - self.u[1:self.n+1, 0:self.n-1])/self.dx

        # North and south - slide 14
        un = np.zeros((self.n-1, self.n))
        un[0:self.n,:] = (self.u[2:self.n+1, 1:self.n] + self.u[1:self.n, 1:self.n])/2
        un[self.n, :] = self.u[self.n+1, 1:self.n]
        us = (self.u[1:self.n, 1:self.n] + self.u[0:self.n-1, 1:self.n])/2
        uyn = (self.u[2:self.n+1, 1:self.n] - self.u[1:self.n, 1:self.n])/self.dx

        stop = True        


if __name__ == "__main__":
    n = 3
    Re = 3
    

    field = CFDSim(n, Re)

    print("stop")
