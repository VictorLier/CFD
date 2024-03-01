import numpy as np
import scipy.sparse as sps

n = 9

AP = np.arange(1,n+1)
AS = np.arange(10,n+10)
AW = np.arange(20,n+20)
AN = np.arange(30,n+30)
AE = np.arange(40,n+40)

D = sps.spdiags([AP,AE,AN,AS,AW],[0,-3,-1,1,3],n,n).T

print(D.toarray())