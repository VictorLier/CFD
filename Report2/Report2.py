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
        self.dy = 1/self.n
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
        self.v[:,-1] = 0


    def NS2dHfunctions(self):
        # U-Grid
        # East and west - slide 14
        ue = np.zeros((self.n, self.n-1))
        ue = (self.u[1:self.n+1, 2:self.n+1] + self.u[1:self.n+1, 1:self.n])/2
        uxe = np.zeros((self.n, self.n-1))
        uxe = (self.u[1:self.n+1, 2:self.n+1] - self.u[1:self.n+1, 1:self.n])/self.dx

        uw = np.zeros((self.n, self.n-1))
        uw = (self.u[1:self.n+1, 1:self.n] + self.u[1:self.n+1, 0:self.n-1])/2
        uxw = np.zeros((self.n, self.n-1))
        uxw = (self.u[1:self.n+1, 1:self.n] - self.u[1:self.n+1, 0:self.n-1])/self.dx


        # North and south - slide 14
        un = np.zeros((self.n, self.n-1))
        un[0:self.n-1,:] = (self.u[2:self.n+1, 1:self.n] + self.u[1:self.n, 1:self.n])/2
        un[self.n-1, :] = self.u[self.n+1, 1:self.n]
        uyn = np.zeros((self.n, self.n-1))
        uyn[0:self.n-1,:] = (self.u[2:self.n+1, 1:self.n] - self.u[1:self.n, 1:self.n])/self.dx
        uyn[self.n-1, :] = 2/self.dx * (self.u[self.n+1, 1:self.n] - self.u[self.n, 1:self.n])

        uvn = (self.v[1:self.n+1, 2:self.n+1] + self.v[1:self.n+1, 1:self.n])/2

        us = np.zeros((self.n, self.n-1))
        us[1:self.n,:] = (self.u[2:self.n+1, 1:self.n] + self.u[1:self.n, 1:self.n])/2 # NB - Lidt funky
        us[0, :] = self.u[0, 1:self.n]
        uys = np.zeros((self.n, self.n-1))
        uys[1:self.n,:] = (self.u[2:self.n+1, 1:self.n] - self.u[1:self.n, 1:self.n])/self.dx
        uys[0, :] = 2/self.dx * (self.u[1, 1:self.n] - self.u[0, 1:self.n])

        uvs = (self.v[0:self.n, 2:self.n+1] + self.v[0:self.n, 1:self.n])/2

        # H1 function - slide 9
        self.H1 = np.zeros((self.n+2, self.n+1))
        self.H1[1:-1,1:-1] = 1 / self.dx * (1 / self.Re * uxe - ue**2) - 1/self.dx * (1 / self.Re * uxw - uw**2) + 1 / self.dy * (1 / Re * uyn - un*uvn) - 1/self.dy * (1 / Re * uys - us*uvs)



        # V-Grid
        # North and south
        vn = np.zeros((self.n-1, self.n))
        vn = (self.v[2:self.n+1, 1:self.n+1] + self.v[1:self.n, 1:self.n+1])/2
        vyn = np.zeros((self.n-1, self.n))
        vyn = (self.v[2:self.n+1, 1:self.n+1] - self.v[1:self.n, 1:self.n+1])/self.dx

        vs = np.zeros((self.n-1, self.n))
        vs = (self.v[1:self.n, 1:self.n+1] + self.v[0:self.n-1, 1:self.n+1])/2
        vys = np.zeros((self.n-1, self.n))
        vys = (self.v[1:self.n, 1:self.n+1] - self.v[0:self.n-1, 1:self.n+1])/self.dx

        # East and west
        ve = np.zeros((self.n-1, self.n))
        ve[:,0:self.n-1] = (self.v[1:self.n, 2:self.n+1] + self.v[1:self.n, 1:self.n])/2
        ve[:,self.n-1] = self.v[1:self.n, self.n+1]
        vxe = np.zeros((self.n-1, self.n))
        vxe[:,0:self.n-1] = (self.v[1:self.n, 2:self.n+1] - self.v[1:self.n, 1:self.n])/self.dx
        vxe[:,self.n-1] = 2/self.dx * (self.v[1:self.n, self.n+1] - self.v[1:self.n, self.n])

        vue = (self.u[2:self.n+1, 1:self.n+1] + self.u[1:self.n, 1:self.n+1])/2

        vw = np.zeros((self.n-1, self.n))
        vw[:,1:self.n] = (self.v[1:self.n, 2:self.n+1] + self.v[1:self.n, 1:self.n])/2 # NB - Lidt funky
        vw[:,0] = self.v[1:self.n, 0]
        vxw = np.zeros((self.n-1, self.n))
        vxw[:,1:self.n] = (self.v[1:self.n, 2:self.n+1] - self.v[1:self.n, 1:self.n])/self.dx
        vxw[:,0] = 2/self.dx * (self.v[1:self.n, 1] - self.v[1:self.n, 0])

        vuw = (self.u[2:self.n+1, 0:self.n] + self.u[1:self.n, 0:self.n])/2

        self.H2 = np.zeros((self.n+1, self.n+2))
        self.H2[1:-1,1:-1] = 1 / self.dx * (1 / self.Re * vxe - ve*vue) - 1/self.dx * (1 / self.Re * vxw - vw*vuw) + 1 / self.dy * (1 / Re * vyn - vn**2) - 1/self.dy * (1 / Re * vys - vs**2)

        stop = True        


if __name__ == "__main__":
    n = 7
    Re = 3
    

    field = CFDSim(n, Re)

    print("stop")
