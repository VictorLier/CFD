import numpy as np
import scipy.sparse as sps

n = 9

AP = np.arange(1,n+1)
AS = np.arange(1,n+1)
AW = np.arange(1,n+1)
AN = np.arange(1,n+1)
AE = np.arange(1,n+1)

D = sps.diags(AP,-1)


print(D.toarray())





