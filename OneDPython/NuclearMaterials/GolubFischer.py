import numpy as np

# This module is the encapsulation of the algorithm
# depicted in:
#
# [1] Golub G.H. "How to generate unknown orthogonal
#     polynomials out of known orthogonal polynomials",
#     Numerical Analysis Project, Stanford University,
#     November 1991.
#
# Comptuting roots of the polynomial is an adaption of
# Newton's method described in:
#
# [2] Barrera-Figueroa V., et al. "Multiple root finder
#     algorithm for Legendre and Chebyshev polynomials
#     via Newton’s method", Annales Mathematicae et
#     Informaticae, volume 33, pages 3-13, 2006.
#
# Finally the weights of the resulting Gauss quadrature
# is obtained as described in:
#
# [3] Sloan D.P., "A New Multigroup Monte Carlo
#     Scattering Algorithm Suitable for Neutral
#     and Charged-Particle Boltzmann and
#     Fokker-Planck Calculations", SAND83-7094,
#     PhD Dissertation, May 1983.
#
#
#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modified Chebyshev Algorithm (MCA) [1]. This function
# accepts as input the recurrence coefficients for
# known orthogonal polynomials of the form:
#
# c_j P_j(x) = (x-a_j)P_{j-1}(x) - b_j P_{j-2}(x)
# P_{-1}(x) = 0, P_0(x) = P_0
#
# As well as 2n-1 amount of moments
# The amount of moments must be an even amount
def MCA(Mell,a,b,c):
    N = np.size(Mell)-1  #This is 2n-1 = L
    n = int((N+1)/2)     #This is n = (L+1)/2

    alpha = np.zeros(n+1)
    beta  = np.zeros(n+1)

    sigma = np.zeros((n+1,2*n+1))

    for ell in range(0,2*n):
        sigma[0,ell] = Mell[ell]

    alpha[0] = a[0] + c[0]*sigma[0,1]/sigma[0,0]
    beta[0]  = Mell[0]

    for k in range(1,n+1):
        for ell in range(k, 2 * n-k+1):
            sigmakm2ell = 0.0
            if k==1:
                sigmakm2ell = 0
            else:
                sigmakm2ell = sigma[k-2,ell]

            sigma[k, ell] = c[ell]*sigma[k-1,ell+1]\
                            - (alpha[k-1]-a[ell])*sigma[k-1,ell] \
                            - beta[k-1]*sigmakm2ell \
                            + b[ell]*sigma[k-1,ell-1]

        alpha[k] = a[k] \
                   - c[k-1]*(sigma[k-1,k]/sigma[k-1,k-1]) \
                   + c[k]*(sigma[k,k+1]/sigma[k,k])
        beta[k] = c[k-1]*sigma[k,k]/sigma[k-1,k-1]

    return alpha,beta

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This function performs a recurrence operation
# to evaluate the reccurence poly order ell at
# x, given the recurrence coefficients
def Ortho(ell,x,alpha,beta):
    if ell==0:
        return 1

    if ell==1:
        return (x-alpha[0])*1 - beta[0]*0

    Pnm1 = 1
    Pn   = (x-alpha[0])*1 - 1*0
    Pnp1 = 0

    for n in range(2,ell+1):
        ns=n-1
        Pnp1 = (x-alpha[ns])*Pn -beta[ns]*Pnm1;
        Pnm1 = Pn;
        Pn = Pnp1;

    return Pnp1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This function performs a recurrence operation
# to evaluate the derivative of
# reccurence poly order ell at
# x, given the recurrence coefficients
def dOrtho(ell,x,alpha,beta):
    eps = 0.000001
    y2 = Ortho(ell,x+eps,alpha,beta)
    y1 = Ortho(ell,x-eps,alpha,beta)

    m = (y2-y1)/2/eps

    return m

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This function computes the roots of the N-th
# order recurring poly given the recurrence
# coefficients [2]. It also computes the weights
# associated with the Gauss-quadrature for this
# polynomial [3].
def RootsOrtho(N,alpha,beta,maxiters=1000,tol=1.0e-10):
    xn = np.linspace(-0.999, 0.999, N);  # Initial guessed values

    wn = np.zeros((N));

    # ======================= Find norm constants
    # Eq. B19b of Sloan
    norm = np.zeros(N + 1)
    norm[0] = beta[0]
    for i in range(1, N + 1):
        norm[i] = beta[i] * norm[i - 1]


    for k in range(0, N):
        i = 0;
        while (i < maxiters):
            xold = xn[k]
            a = Ortho(N, xold,alpha,beta)
            b = dOrtho(N, xold,alpha,beta)
            c = 0;
            for j in range(0, k):
                c = c + (1 / (xold - xn[j]))

            xnew = xold - (a / (b - a * c))

            res = abs(xnew - xold)
            xn[k] = xnew

            if (res < tol):
                break
            i = i + 1

    # ======================Sorting the roots
    for i in range(0, N - 1):
        for j in range(0, N - i - 1):
            if (xn[j] > xn[j + 1]):
                tempx = xn[j + 1]
                tempw = wn[j + 1]
                xn[j + 1] = xn[j]
                wn[j + 1] = wn[j]
                xn[j] = tempx
                wn[j] = tempw

    # ======================= Find weights
    # Eq. B2 of Sloan
    for i in range(0, N):
        wn[i] = 0.0
        for k in range(0, N):
            wn[i] += Ortho(k, xn[i], alpha, beta) * \
                     Ortho(k, xn[i], alpha, beta) / norm[k]

        wn[i] = 1.0 / wn[i]
    return xn, wn

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main function
# Computes discrete scattering angles and
# probabilities for sampling angular scattering.
def GetDiscreteScatAngles(Mell):
    N = np.size(Mell) - 1  # This is 2n-1 = L
    n = int((N + 1) / 2)  # This is n = (L+1)/2

    # ================== Legendre recurrence coefficients
    a = np.zeros(2 * n)
    b = np.zeros(2 * n)
    c = np.zeros(2 * n)

    for j in range(0, 2 * n):
        a[j] = 0.0
        b[j] = j / (2 * j + 1)
        c[j] = (j + 1) / (2 * j + 1)

    # ================== Find orthogonal poly recur. coeffs
    alpha, beta = MCA(Mell, a, b, c)

    # ======================= Find roots
    xn, wn = RootsOrtho(n, alpha, beta)

    return xn, wn