import numpy as np
import matplotlib.pyplot as plt

N=20
x = np.linspace(-1.0,1.0,N)

A = np.zeros((N,N))
b = np.zeros(N)
veca = np.zeros(N)
vecb = np.zeros(N)
vecc = np.zeros(N)
vecd = np.zeros(N)


## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc

for i in range(0,N):

    D_g = 3.0
    dx = 2/N
    dx2=dx*dx
    sigma_r = 0.9

    a_e = 0.0
    a_w = 0.0
    a_c = 1.0
    if (i == 0):
        a_e = -1.0 * D_g / dx2
        a_w = 0.0
        a_c = 0.5 / dx + sigma_r - a_e
        A[i, i + 1] = a_e
        vecc[0] = a_e
    elif (i == (N - 1)):
        a_e = 0.0
        a_w = -1.0 * D_g / dx2
        a_c = 0.5 / dx + sigma_r - a_w
        A[i, i - 1] = a_w
        veca[i-1] = a_w
    else:
        a_e = -1.0 * D_g / dx2
        a_w = -1.0 * D_g / dx2
        a_c = sigma_r - a_e - a_w
        A[i, i + 1] = a_e
        A[i, i - 1] = a_w
        veca[i-1] = a_w
        vecc[i] = a_e

    A[i, i] = a_c
    vecb[i]=a_c

    b[i] = 1.0
    vecd[i] = 1.0

phi = np.linalg.solve(A,b)
phi2 = TDMAsolver(veca,vecb,vecc,vecd)

plt.plot(x,phi)
plt.plot(x,phi2)
plt.show()

