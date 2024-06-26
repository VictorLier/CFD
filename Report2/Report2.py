import numpy as np
import scipy as sp
from typing import List, Callable
import sympy as sp
from scipy import interpolate
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = "12"
plt.rcParams["font.family"] = "serif"
import scipy.sparse as sps
import time
import multiprocessing

class CFDSim:
    def __init__(self, _n, _Re, Ulid = 1, maxstep = 100000, dt = None, steadytol = 10e-3, div_correction:bool=False) -> None:
        self.n = _n
        self.Re = _Re
        self.Ulid = Ulid
        self.maxstep = maxstep
        self.StaggeredMesh2d()
        self.dt = self.get_dt(dt)
        self.steadytol = steadytol
        self.p = np.zeros((self.n, self.n))
        self.gmchist = np.zeros(self.maxstep)
        self.cmchist = np.zeros(self.maxstep)
        self.steadyhist = np.zeros(self.maxstep)

        self.div_correction = div_correction

        self.u = None
        self.v = None

        self.A = None
        self.LU = None
        self.s = None

    def get_dt(self, dt):
        dt_min = np.min([self.Re * self.dx**2 / 4, 1 / (self.Ulid**2 * self.Re)])/2
        if dt is None:
            return dt_min
        else:
            return min(dt, dt_min)

    def StaggeredMesh2d(self):
        self.dx = 1/self.n   # cell size in x,y
        self.dy = 1/self.n
        self.xf_ = np.arange(0, 1 + self.dx, self.dx)   # cell face coordinate vector, 1D
        self.xc_ = np.arange(self.dx/2, 1 - self.dx/2, self.dx)  # cell center coordinate vector, 1D
        self.xf = [p*self.dx for p in range(self.n+1)]   # cell face coordinate vector, 1D
        self.xc = [self.dx/2 + p * self.dx for p in range(self.n)]  # cell center coordinate vector, 1D
        self.xb = np.zeros(self.n+2)
        self.xb[0] = 0
        self.xb[-1] = 1
        self.xb[1:-1] = self.xc     # cell center coordinate vector incl boundaries
        self.Xu, self.Yu = np.meshgrid(self.xf, self.xb)    # u-grid coordinate array
        self.Xv, self.Yv = np.meshgrid(self.xb, self.xf)    # v-grid coordinate array
        self.Xp, self.Yp = np.meshgrid(self.xc, self.xc)    # p-grid coordinate array
        self.Xi, self.Yi = np.meshgrid(self.xf, self.xf)    # fd-grid coordinate array
    
    def LidDrivenCavity(self):
        self.u = np.zeros((self.n+2, self.n+1))
        # self.u[1:-1,:] = self.Xu[1:-1,:] * (1 - self.Xu[1:-1,:])
        self.u[-1,:] = self.Ulid
        self.v = np.zeros((self.n+1, self.n+2))
        # self.v[:,1:-1] = self.Yv[:,1:-1] * (1 - self.Yv[:,1:-1])
        self.v[:,-1] = 0

    def NS2dHfunctions(self, U: Callable | None = None, V: Callable | None = None, Re = None):
        if self.u is not None or self.v is not None:
            pass
        elif U is None or V is None:
            self.LidDrivenCavity()
        else:
            self.u = U(self.Xu, self.Yu)
            self.v = V(self.Xv, self.Yv)

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
        self.H1[1:-1,1:-1] = 1 / self.dx * (1 / self.Re * uxe - ue**2) - 1/self.dx * (1 / self.Re * uxw - uw**2) + 1 / self.dy * (1 / self.Re * uyn - un*uvn) - 1/self.dy * (1 / self.Re * uys - us*uvs)

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
        self.H2[1:-1,1:-1] = 1 / self.dx * (1 / self.Re * vxe - ve*vue) - 1/self.dx * (1 / self.Re * vxw - vw*vuw) + 1 / self.dy * (1 / self.Re * vyn - vn**2) - 1/self.dy * (1 / self.Re * vys - vs**2)
        return self.H1, self.H2
    
    def NS2LaplaceMatrix(self):
        # Assuming dx = dy:
        self._a_n = np.full([self.n, self.n], 1)
        self._a_s = np.full([self.n, self.n], 1)
        self._a_e = np.full([self.n, self.n], 1)
        self._a_w = np.full([self.n, self.n], 1)
        self._a_p = -(self._a_n + self._a_s + self._a_e + self._a_w)

        # North
        self._a_p[-1,:] =  self._a_p[-1,:] + self._a_n[-1,:]
        self._a_n[-1,:] = 0

        # South
        self._a_p[0,:] =  self._a_p[0,:] + self._a_s[0,:]
        self._a_s[0,:] = 0

        # East
        self._a_p[:,-1] =  self._a_p[:,-1] + self._a_e[:,-1]
        self._a_e[:,-1] = 0

        # West
        self._a_p[:,0] =  self._a_p[:,0] + self._a_w[:,0]
        self._a_w[:,0] = 0

        data = np.array([self._a_p.flatten('F'), self._a_e.flatten('F'), self._a_n.flatten('F'), self._a_s.flatten('F'), self._a_w.flatten('F')])

        self.A = sps.spdiags(data, [0, -self.n, -1, 1, self.n], self.n*self.n, self.n*self.n, format = 'csr').T #NB MAYBE USE CSC
        return self.A
    
    def get_source(self, H1 = None, H2 = None):
        if H1 is not None or H2 is not None:
            self.H1 = H1
            self.H2 = H2

        if self.div_correction: # If divergence correction is enabled
            H1_c = 1 / self.dt * self.u + self.H1
            H2_c = 1 / self.dt * self.v + self.H2

            H1w = H1_c[1:-1,:-1]
            H1e = H1_c[1:-1,1:]
            H2s = H2_c[:-1,1:-1]
            H2n = H2_c[1:,1:-1]

        else:
            H1w = self.H1[1:-1,:-1]
            H1e = self.H1[1:-1,1:]
            H2s = self.H2[:-1,1:-1]
            H2n = self.H2[1:,1:-1]

        s = (H1e - H1w) * self.dy + (H2n - H2s) * self.dx

        # # North
        # s[-1,:] = s[-1,:] - self.dy*self._a_n[-1,:]*self.H2[self.n-1,1:-1]
        # # South
        # s[0,:] = s[0,:] + self.dy*self._a_s[0,:]*self.H2[0,1:-1]
        # # East
        # s[:,-1] = s[:,-1] - self.dx*self._a_e[:,-1]*self.H1[1:-1,self.n-1]
        # # West
        # s[:,0] = s[:,0] + self.dx*self._a_w[:,0]*self.H1[1:-1,0]
        self.s = s
        return self.s


    def PoisonSolver(self, H1 = None, H2 = None):
        if self.A is None:
            self.NS2LaplaceMatrix()
        self.get_source(H1, H2)
        _p = sps.linalg.spsolve(self.A, self.s.flatten()).reshape(self.n, self.n)
        self.p = _p.copy() - _p[np.ceil(self.n/2-1).astype(int), np.ceil(self.n/2-1).astype(int)]
        # self.p = _p.copy() - np.mean(_p)
        return self.p

    def LU_PoisonSolver(self, H1 = None, H2 = None):
        if self.A is None or self.LU is None:
            self.NS2LaplaceMatrix()
            self.LU = sps.linalg.splu(self.A)

        self.get_source(H1, H2)

        _p = self.LU.solve(self.s.flatten()).reshape(self.n, self.n)

        self.p = _p.copy() - _p[np.ceil(self.n/2-1).astype(int), np.ceil(self.n/2-1).astype(int)]
        # self.p = _p.copy() - np.mean(_p)
        return self.p

    def NS2dMovingLidSquareCavityFlow(self, plot = False, LU_optimization = False):
        self.NS2LaplaceMatrix()
        self.LidDrivenCavity()
        u_step = self.u.copy()
        v_step = self.v.copy()
        steadytest = 1
        step = 0
        while steadytest > self.steadytol and step < self.maxstep:
            step += 1
            self.NS2dHfunctions()
            if LU_optimization:
                self.LU_PoisonSolver()
            else:
                self.PoisonSolver()

            dudt = self.H1[1:-1,1:-1] - 1 / self.dx * (self.p[:,1:] - self.p[:,:-1])
            dvdt = self.H2[1:-1,1:-1] - 1 / self.dy * (self.p[1:,:] - self.p[:-1,:])

            u_step[1:-1, 1:-1] = self.u[1:-1, 1:-1] + self.dt * dudt
            v_step[1:-1, 1:-1] = self.v[1:-1, 1:-1] + self.dt * dvdt

            self.u = u_step.copy()
            self.v = v_step.copy()

            dudx = (self.u[1:-1, 1:] - self.u[1:-1, :-1]) / self.dx
            dvdy = (self.v[1:, 1:-1] - self.v[:-1, 1:-1]) / self.dy

            self.cmchist[step-1] = np.max(np.abs(dudx + dvdy))
            self.gmchist[step-1] = np.sum(np.abs(dudx + dvdy))

            steadytest = np.max(np.abs(dudt)) + np.max(np.abs(dvdt))

            progress = self.steadytol/steadytest*100
            # if step % 1000 == 0:
            print(f"n = {self.n} - Step: {step}/{self.maxstep} - Progress: {progress:.3f}%")

        # Convert to velocities to the p-grid
        u_p = np.zeros((self.n + 2, self.n + 2))
        v_p = np.zeros((self.n + 2, self.n + 2))
        u_p[:, 1:-1] = (self.u[:,1:] + self.u[:,:-1])/2
        v_p[1:-1, :] = (self.v[1:,:] + self.v[:-1,:])/2

        if plot:
            # Show the velocity field using streamlines
            plt.figure(figsize=(4, 4))
            plt.streamplot(self.Xp, self.Yp, u_p[1:-1,1:-1], v_p[1:-1,1:-1])
            # plt.title(f"Streamlines, n = {self.n}, Re = {self.Re}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.tight_layout()
            plt.savefig(f"Streamlines_n{self.n}_Re{self.Re}.pdf")


            # Make a contour plot of the pressure field
            plt.figure(figsize=(5, 4))  # Adjust width to accommodate the colorbar
            plt.subplot(111)  # Subplot for the main plot

            # Plot the pressure field
            plt.pcolormesh(self.Xp, self.Yp, self.p)

            # Add colorbar to the right of the plot
            cbar = plt.colorbar(label='Pressure')
            cbar.ax.set_ylabel('Pressure', rotation=270, labelpad=15)  # Rotate label for better fit

            # Set title and axis labels
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim((0, 1))
            plt.ylim((0, 1))

            plt.tight_layout()

            # Adjust the width of the colorbar to match the plot
            plt.subplots_adjust(right=0.85)

            plt.savefig(f"Pressure_n{self.n}_Re{self.Re}.pdf")


            plt.figure()
            plt.plot(np.arange(step), self.gmchist[:step], label="Global Mass Conservation")
            plt.plot(np.arange(step), self.cmchist[:step], label="Local Mass Conservation")
            plt.title("Mass Conservation")
            plt.xlabel("Step")
            plt.ylabel("Conservation")
            plt.legend()
            plt.grid()

            np.savetxt("GlobalMassConservation.txt", np.array([np.arange(step), self.gmchist[:step]]).T)
            np.savetxt("ContinuityMassConservation.txt", np.array([np.arange(step), self.cmchist[:step]]).T)


class testCFDSim:
    def __init__(self, N: list, Re: float, K: list[float, float, float, float]) -> None:
        self.N = N
        self.Re = Re
        self.K = K
        self.fields = self.init_fields()
    
    def init_fields(self):
        return [CFDSim(n, self.Re) for n in self.N]

    def NS2dValidateHfunction(self):
        x,y = sp.symbols('x y')
        _U = sp.sin(self.K[0]*x) * sp.sin(self.K[1]*y)
        _V = sp.sin(self.K[2]*x) * sp.sin(self.K[3]*y)

        _Ux = sp.diff(_U, x)
        _Uy = sp.diff(_U, y)

        _Vx = sp.diff(_V, x)
        _Vy = sp.diff(_V, y)

        _H1e = sp.diff(1/self.Re * _Ux - _U**2, x) + sp.diff(1/self.Re * _Uy - _U*_V, y)
        _H2e = sp.diff(1/self.Re * _Vx - _V*_U, x) + sp.diff(1/self.Re * _Vy - _V**2, y)

        U = sp.lambdify((x, y), _U)
        V = sp.lambdify((x, y), _V)

        H1e = sp.lambdify((x, y), _H1e)
        H2e = sp.lambdify((x, y), _H2e)

        H1Error = np.zeros(len(self.fields))
        H2Error = np.zeros(len(self.fields))

        for i, f in enumerate(self.fields):
            H1, H2 = f.NS2dHfunctions(U, V, self.Re)
            Xu, Yu = f.Xu, f.Yu
            Xv, Yv = f.Xv, f.Yv
            H1exact = H1e(Xu, Yu)
            H2exact = H2e(Xv, Yv)
            H1Error[i] = np.max(abs(H1[1:-1,1:-1] - H1exact[1:-1,1:-1]))
            H2Error[i] = np.max(abs(H2[1:-1,1:-1] - H2exact[1:-1,1:-1]))

        slopeH1 = np.polyfit(np.log(self.N),np.log(H1Error),1)[0]
        slopeH2 = np.polyfit(np.log(self.N),np.log(H2Error),1)[0]

        plt.figure()
        plt.loglog(self.N, H1Error, label=f"Slope H1: {slopeH1:.2f}", marker="o")
        plt.loglog(self.N, H2Error, label=f"Slope H2: {slopeH2:.2f}", marker="o")
        plt.legend()
        plt.grid()

        # print to latex
        for i, n in enumerate(self.N):
            print(f"{n} {H1Error[i]} ")
        
        for i, n in enumerate(self.N):
            print(f"{n} {H2Error[i]} ")

    def NS2dValidatePoissonSolver(self, K):
        H1e = lambda x,y: -K * np.sin(K*x)*np.cos(K*y)
        H2e = lambda x,y: -K * np.cos(K*x)*np.sin(K*y)
        Pe = lambda x,y: np.cos(K*x)*np.cos(K*y)

        PError = np.zeros(len(self.fields))

        for i, f in enumerate(self.fields):
            Xu, Yu = f.Xu, f.Yu
            Xv, Yv = f.Xv, f.Yv
            Xp, Yp = f.Xp, f.Yp
            H1exact = H1e(Xu, Yu)
            H2exact = H2e(Xv, Yv)       
            P = f.PoisonSolver(H1exact, H2exact)
            Pexact = Pe(Xp, Yp)
            Pmean = P - np.mean(P) 
            # Pmean = P - P[np.ceil(f.n/2).astype(int), np.ceil(f.n/2).astype(int)]+1
            PError[i] = np.max(abs(Pmean - Pexact))
        
        slopeP = np.polyfit(np.log(self.N),np.log(PError),1)[0]

        plt.figure()
        plt.loglog(self.N, PError, label=f"Slope P: {slopeP:.2f}", marker="o")
        plt.legend()
        plt.grid()
        # print to latex
        for i, n in enumerate(self.N):
            print(f"{n} {PError[i]} ")

def Convergence(N):
    Y = np.array([[ 1.0000, -1.00000,  0.052971],
            [ 0.9688, -0.58031,  0.051493],
            [ 0.9531, -0.47239,  0.050314],
            [ 0.7344, -0.18861,  0.012113],
            [ 0.5000,  0.06205,  0.000000],
            [ 0.2813,  0.28040,  0.040381],
            [ 0.1016,  0.30029,  0.104416],
            [ 0.0625,  0.20227,  0.109160],
            [ 0.0000,  0.00000,  0.110560]])

    X = np.array([[ 0.0000,  0.00000,  0.077429],
        [ 0.0391, -0.29330,  0.078658],
        [ 0.0547, -0.41018,  0.077128],
        [ 0.1406, -0.42634,  0.049004],
        [ 0.5000,  0.02580,  0.000000],
        [ 0.7734,  0.33398,  0.047259],
        [ 0.9062,  0.33290,  0.084369],
        [ 0.9297,  0.29627,  0.087625],
        [ 1.0000,  0.00000,  0.090448]])

    Errors = np.zeros([4])
    sim = CFDSim(_n = N, _Re = 1000, dt=0.01, Ulid=-1, div_correction=True, steadytol=10e-6, maxstep=100000000)
    sim.NS2dMovingLidSquareCavityFlow(plot = False, LU_optimization=True)
    Py = sim.p[:,int(N/2)]
    Px = sim.p[int(N/2), :]
    V = (sim.v[int(np.floor(N/2)),:] + sim.v[int(np.ceil(N/2)),:]) / 2
    U = (sim.u[:,int(np.floor(N/2))] + sim.u[:,int(np.ceil(N/2))]) / 2
    Py_values = interpolate.interp1d(sim.xc, Py, kind='cubic', fill_value='extrapolate')
    Px_values = interpolate.interp1d(sim.xc, Px, kind='cubic', fill_value='extrapolate')
    V_values = interpolate.interp1d(sim.xb, V, kind='cubic', fill_value='extrapolate')
    U_values = interpolate.interp1d(sim.xb, U, kind='cubic', fill_value='extrapolate')

    Errors[0] = np.max(np.abs(Py_values(Y[:,0]) - Y[:,2]))
    Errors[1] = np.max(np.abs(Px_values(X[:,0]) - X[:,2]))
    Errors[2] = np.max(np.abs(V_values(X[:,0]) - X[:,1]))
    Errors[3] = np.max(np.abs(U_values(Y[:,0]) - Y[:,1]))
    np.savetxt(f"Report2/Convergence_files/Errors_{N}.txt", Errors)

    # Plot the benchmark values together with the computed values in four subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(Y[:,0], Y[:,2], 'o-', label='Benchmark')
    axs[0, 0].plot(Y[:,0], Py_values(Y[:,0]), 'o-', label='Computed')
    axs[0, 0].set_title('Pressure along y-axis')
    axs[0, 0].set_xlabel('y')
    axs[0, 0].set_ylabel('Pressure')
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(X[:,0], X[:,2], 'o-', label='Benchmark')
    axs[0, 1].plot(X[:,0], Px_values(X[:,0]), 'o-', label='Computed')
    axs[0, 1].set_title('Pressure along x-axis')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('Pressure')
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 0].plot(X[:,0], X[:,1], 'o-', label='Benchmark')
    axs[1, 0].plot(X[:,0], V_values(X[:,0]), 'o-', label='Computed')
    axs[1, 0].set_title('Velocity along x-axis')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('Velocity')
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(Y[:,0], Y[:,1], 'o-', label='Benchmark')
    axs[1, 1].plot(Y[:,0], U_values(Y[:,0]), 'o-', label='Computed')
    axs[1, 1].set_title('Velocity along y-axis')
    axs[1, 1].set_xlabel('y')
    axs[1, 1].set_ylabel('Velocity')
    axs[1, 1].grid()
    axs[1, 1].legend()
    fig.tight_layout()
    plt.savefig(f"Report2/Convergence_files/Convergence_{N}.pdf")

    # print to latex
    print('Y Pressure')
    for i in range(len(Y)):
        print(f"{Y[i,0]} {Y[i,2]} {Py_values(Y[i,0])}")
    print('X Pressure')
    for i in range(len(X)):
        print(f"{X[i,0]} {X[i,2]} {Px_values(X[i,0])}")

    print('X Velocity')
    for i in range(len(X)):
        print(f"{X[i,0]} {X[i,1]} {V_values(X[i,0])}")

    print('Y Velocity')
    for i in range(len(Y)):
        print(f"{Y[i,0]} {Y[i,1]} {U_values(Y[i,0])}")

    return Errors


if __name__ == "__main__":
    if False: # Test
        sim = CFDSim(3, 10)
        sim.NS2dHfunctions()
        sim.NS2LaplaceMatrix()
        sim.PoisonSolver()

        print("stop")

    if False: # Question 2
        N = np.round(np.logspace(1, 3, 10)).astype(int)
        K = np.array([1, 2, 3, 4]) * np.pi
        Re = 3

        test = testCFDSim(N, Re, K)
        test.NS2dValidateHfunction()

    if False: # Question 3
        N = np.round(np.logspace(1, 3, 30)).astype(int)
        # N = np.array([9, 20, 40, 80])
        K = np.array([1, 2, 3, 4]) * np.pi
        Re = 3

        test = testCFDSim(N, Re, K)
        test.NS2dValidatePoissonSolver(2*np.pi)
    
    if False: # Question 5
        sim = CFDSim(_n = 21, _Re = 1, dt=None, Ulid=-1)
        sim.NS2dMovingLidSquareCavityFlow(plot = True)

    if True: # Question 6
        sim = CFDSim(_n = 21, _Re = 1, dt=None, Ulid=-1, div_correction=True)
        sim.NS2dMovingLidSquareCavityFlow(plot = True)

    if False: # Question 7
    
        n_arr = np.array([11])
        sims = np.zeros(len(n_arr), dtype=object)
        t_arr = np.zeros([len(n_arr), 2])

        for i, n in enumerate(n_arr):
            sims[i] = CFDSim(_n = n, _Re = 1, dt=0.01, Ulid=-1, div_correction=True, steadytol=10e-6)

        for i in range(len(n_arr)):
            start = time.time()
            sims[i].NS2dMovingLidSquareCavityFlow(plot = False, LU_optimization=True)
            end = time.time()
            t_arr[i, 0] = end - start

        for i in range(len(n_arr)):
            start = time.time()
            sims[i].NS2dMovingLidSquareCavityFlow(plot = False, LU_optimization=False)
            end = time.time()
            t_arr[i, 1] = end - start

        slope_LU = np.polyfit(np.log(n_arr),np.log(t_arr[:,0]),1)[0]
        slope_noLU = np.polyfit(np.log(n_arr),np.log(t_arr[:,1]),1)[0]

        plt.figure()
        plt.loglog(n_arr, t_arr[:,0], label=f"Slope LU: {slope_LU:.2f}", marker="o")
        plt.loglog(n_arr, t_arr[:,1], label=f"Slope No LU: {slope_noLU:.2f}", marker="o")
        plt.legend()
        plt.grid()

        np.savetxt("LUOptimization.txt", np.array([n_arr, t_arr[:,0]]).T)
        np.savetxt("NoLUOptimization.txt", np.array([n_arr, t_arr[:,1]]).T)

    if False: # Question 7B
        # n_arr = np.array([21, 51, 101])
        n_arr = np.logspace(1, 3, 10).astype(int)
        sims = np.zeros(len(n_arr), dtype=object)
        t_arr = np.zeros([len(n_arr), 2])
        n_test = 100

        for i, n in enumerate(n_arr):
            sims[i] = CFDSim(_n = n, _Re = 1, dt=0.01, Ulid=-1, div_correction=True, steadytol=10e-6)

        for i in range(len(n_arr)):
            print(f"LU: n: {n_arr[i]}")
            sim = sims[i]
            sim.NS2LaplaceMatrix()
            sim.LidDrivenCavity()
            sim.NS2dHfunctions()
            LU_tests = np.zeros(n_test)
            for j in range(n_test):
                start = time.perf_counter()
                sim.LU_PoisonSolver()
                end = time.perf_counter()
                LU_tests[j] = end - start
            t_arr[i, 0] = np.median(LU_tests)

        for i in range(len(n_arr)):
            print(f"No LU: n: {n_arr[i]}")
            sim = sims[i]
            sim.NS2LaplaceMatrix()
            sim.LidDrivenCavity()
            sim.NS2dHfunctions()
            NoLU_tests = np.zeros(n_test)
            
            for j in range(n_test):
                start = time.perf_counter()
                sim.PoisonSolver()
                end = time.perf_counter()
                NoLU_tests[j] = end - start
            t_arr[i, 1] = np.median(NoLU_tests)

        slope_LU = np.polyfit(np.log(n_arr),np.log(t_arr[:,0]),1)[0]
        slope_noLU = np.polyfit(np.log(n_arr),np.log(t_arr[:,1]),1)[0]

        np.savetxt(f"Report2/Timing_files/LUOptimization_slope{slope_LU}.txt", np.array([n_arr, t_arr[:,0]]).T)
        np.savetxt(f"Report2/Timing_files/NoLUOptimization_slope{slope_noLU}.txt", np.array([n_arr, t_arr[:,1]]).T)

        plt.figure()
        plt.loglog(n_arr, t_arr[:,0], label=f"Slope LU: {slope_LU:.2f}", marker="o")
        plt.loglog(n_arr, t_arr[:,1], label=f"Slope No LU: {slope_noLU:.2f}", marker="o")
        plt.legend()
        plt.grid()

    if False: # Question 8
        sim = CFDSim(_n = 21, _Re = 1000, dt=None, Ulid=-1, div_correction=True)
        sim.NS2dMovingLidSquareCavityFlow(plot = True)

        Numbers = [11, 17, 27, 35, 51, 75, 101, 131, 161, 201]
        # Numbers = [51]
        Errors = np.zeros((len(Numbers) + 1, 4))

        if False:
            pool = multiprocessing.Pool(processes=min(len(Numbers),5))
            # pool = multiprocessing.Pool()In 
            for n in Numbers:
                pool.apply_async(Convergence, args=[n])

            pool.close()
            pool.join()

        elif True:
            pass

        else:
            for i, n in enumerate(Numbers):
                Convergence(n)

        for i, n in enumerate(Numbers):
            Errors[i] = np.loadtxt(f"Report2/Convergence_files/Errors_{n}.txt")
    

        slope_Py = np.polyfit(np.log(Numbers),np.log(Errors[:-1,0]),1)[0]
        slope_Px = np.polyfit(np.log(Numbers),np.log(Errors[:-1,1]),1)[0]
        slope_V = np.polyfit(np.log(Numbers),np.log(Errors[:-1,2]),1)[0]
        slope_U = np.polyfit(np.log(Numbers),np.log(Errors[:-1,3]),1)[0]

        Errors[-1, 0] = slope_Py
        Errors[-1, 1] = slope_Px
        Errors[-1, 2] = slope_V
        Errors[-1, 3] = slope_U
        np.savetxt("Errors.txt", Errors)

        plt.figure()
        plt.loglog(Numbers, Errors[:-1,0], label=f"Slope Py: {slope_Py:.2f}", marker="o")
        plt.loglog(Numbers, Errors[:-1,1], label=f"Slope Px: {slope_Px:.2f}", marker="o")
        plt.loglog(Numbers, Errors[:-1,2], label=f"Slope V: {slope_V:.2f}", marker="o")
        plt.loglog(Numbers, Errors[:-1,3], label=f"Slope U: {slope_U:.2f}", marker="o")
        plt.legend()


    plt.show()
