import numpy as np
import math
import matplotlib.pyplot as plt

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
fun = lambda x,y: np.sin(k*x)*np.sin(k*y)
print(fun(0.8,0.8))
dfun = lambda x,y: k*k*np.cos(k*X)*np.cos(k*Y)
D = L/(N-1)
error = np.zeros((1, len(N)))

for i, N in enumerate(N):
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x,y)

    f = fun(X,Y)

    Dx = diffmatrix_1d_general(DERx,X[0,:],a,b) # Lidt Wired med X[0,:]
    Dy = diffmatrix_1d_general(DERy,Y[:,0],a,b)
    print(Dx)

    IdentMat = np.eye(N)

    DX = np.kron(Dx, IdentMat)
    DY = np.kron(IdentMat, Dy)

    print(DX)


    Fx = 1/(2*D)* np.dot(Dx,f)


    D_comb = np.hstack((DX,DY)) # ChatGPT bedste bud

    fun_flat = fun.flatten()
    dfun_flat = 1/(2*D) * np.dot(D_comb,fun_flat)

    dfun_flat_2D = dfun_flat.reshape(fun.shape)


    print(1)