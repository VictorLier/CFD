import numpy as np

from Report1_scripts.FiniteDifference1D import fdcoeff_1d_general, fdcoeff_1d_uniform

N = np.round(np.logspace(np.log10(3), np.log10(1000), 25)).astype(int)

ERROR = np.zeros(len(N))


fun = lambda z: 4*z*(1-z)

for i, n in enumerate(N):
    #A matrix
    A = np.zeros((n,n))
    np.fill_diagonal(A,-2)
    np.fill_diagonal(A[:,1:],1)
    np.fill_diagonal(A[1:],1)
    A[0,0] = 1
    A[0,1] = 0
    A[n-1,n-1] = 1
    A[n-1,n-2] = 0
    
    # B vector
    b_vec = np.full(n,8)
    b_vec[0] = 0
    b_vec[n-1] = 0
    b = - 1/n**2 * b_vec
    
    #Solve u
    u = np.linalg.solve(A,b)

    z = np.linspace(0,1,n)
    u_exact = fun(z)

    ERROR[i] = np.max(abs(u-u_exact))

data = np.vstack((N,ERROR))

print(N)

print(ERROR)

print(data)
