import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sps

from fdcoeff_1d import diffmatrix_1d_general


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
error = np.zeros((1, len(N)))

for i, n in enumerate(N):
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)

    X, Y = np.meshgrid(x,y)

    fun = np.sin(k*X)*np.sin(k*Y)
    dfun = k*k*np.cos(k*X)*np.cos(k*Y)

    Dx = diffmatrix_1d_general(DERx,x,a,b)
    Dy = diffmatrix_1d_general(DERy,y,a,b)

    IdentMat = sps.identity(n)

    DX = sps.kron(Dx, IdentMat)
    DY = sps.kron(IdentMat, Dy)

    D_xy = np.dot(DX,DY)

    fun_flat = fun.flatten()
    dfun_flat = D_xy.dot(fun_flat)

    dfun_flat_2D = dfun_flat.reshape(fun.shape)

    error[0,i] = np.linalg.norm(dfun - dfun_flat_2D, np.inf)
    print(n, error[0,i])

print(error)