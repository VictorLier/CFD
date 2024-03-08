import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sps
import array_to_latex as a2l

from fdcoeff_1d import diffmatrix_1d_general


def fdcoeff_2d_general(DERx, DERy, x, y, a, b):
    Dx = diffmatrix_1d_general(DERx,x,a,b)
    Dy = diffmatrix_1d_general(DERy,y,a,b)

    IdentMat = sps.identity(n)

    DX = sps.kron(Dx, IdentMat)
    DY = sps.kron(IdentMat, Dy)

    D_xy = DX.dot(DY)

    # in a2l.to_ltx(D_xy.toarray(), frmt = '{:6.0f}', arraytype = 'array')

    return D_xy


#Input
N = np.array([20, 40, 80, 160, 320])
L = 1
a = 1
b = 1
DERx = 1
DERy = 1

#Initialize
k = 2*math.pi/L
#fun = math.sin(k*X)*math.sin(k*Y)
#dfun = k*k*math.cos(k*X)*math.cos(k*Y)
D = L/(N-1)
error = np.zeros(len(N))

for i, n in enumerate(N):
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)

    X, Y = np.meshgrid(x,y)

    fun = np.sin(k*X)*np.sin(k*Y)
    dfun = k*k*np.cos(k*X)*np.cos(k*Y)

    D_xy = fdcoeff_2d_general(DERx, DERy, x, y, a, b)

    fun_flat = fun.flatten()
    dfun_flat = D_xy.dot(fun_flat)

    dfun_aprrox = dfun_flat.reshape(fun.shape)

    error[i] = np.abs(dfun_aprrox - dfun).max()

_roc = np.polyfit(np.log(N), np.log(error), 1)
roc = _roc[0]
TL = np.polyval(_roc, np.log(N))

# plot 2 subplots
plt.figure()
plt.suptitle("Convergence test")
plt.loglog(N, error, label=f"Slope: {roc:.2f}", marker="o")
# plt.semilogx(N, TL)
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.grid()

plt.show()