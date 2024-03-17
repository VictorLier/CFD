import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sps
import array_to_latex as a2l

from FiniteDifference1D import diffmatrix_1d_general


def fdcoeff_2d_general(DERx, DERy, x, y, a, b, n):
    Dx = diffmatrix_1d_general(DERx,x,a,b)
    Dy = diffmatrix_1d_general(DERy,y,a,b)

    IdentMat = sps.identity(n)

    DX = sps.kron(Dx, IdentMat)
    DY = sps.kron(IdentMat, Dy)

    D_xy = DX.dot(DY)

    # in a2l.to_ltx(D_xy.toarray(), frmt = '{:6.0f}', arraytype = 'array')

    return D_xy
