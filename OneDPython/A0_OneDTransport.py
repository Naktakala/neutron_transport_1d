import numpy as np
import Legendre as lg
import matplotlib.pyplot as plt
import MonteCarlo1D as mc
import math



def DD(N_a,N_e):
    #============================ Problem physical properties
    a = 0.0
    b = 1.0
    sig_s = 0.0
    sig_t = 1.0
    S = 1.0

    #============================ Problem numerical properties
    tol = 1.0e-6

    #============================ Initialization
    h = (b-a)/N_e
    mu,w = lg.LegendreRoots(N_a)

    psi_Left  = np.zeros((N_a))
    psi_Right = np.zeros((N_a))
    phi_old   = np.zeros((N_e))
    phi_new   = np.zeros((N_e))

    x = np.zeros((N_e))
    for i in range(0,N_e):
        x[i] = a + 0.5*h + i*h

    for iter in range(0,101):
        psi_imh = 0.0
        psi_iph = 0.0
        for n in range(0,N_a):


            #==================== Sweep right
            if (mu[n]>0):
                for i in range(0,N_e):
                    if (i==0):
                        psi_imh = psi_Left[n]
                    else:
                        psi_imh = psi_iph

                    a_e = mu[n] / h + 0.5 * sig_t
                    a_w = mu[n] / h - 0.5 * sig_t
                    b = 0.5 * sig_s * phi_old[i] + 0.5 * S

                    psi_iph = (a_w/a_e)*psi_imh + b/a_e

                    phi_new[i] = phi_new[i] + w[n]*0.5*(psi_iph + psi_imh)

            #==================== Sweep left
            else:
                for i in range((N_e-1),-1,-1):
                    if (i==(N_e-1)):
                        psi_iph = psi_Right[n]
                    else:
                        psi_iph = psi_imh

                    a_e = mu[n] / h + 0.5 * sig_t
                    a_w = mu[n] / h - 0.5 * sig_t
                    b = 0.5 * sig_s * phi_old[i] + 0.5 * S

                    psi_imh = (a_e/a_w)*psi_iph - b/a_w

                    phi_new[i] = phi_new[i] + w[n]*0.5*(psi_iph + psi_imh)

        #=================== Compute convg
        norm_phi_chg = np.linalg.norm(phi_new - phi_old)
        norm_phi_new = np.linalg.norm(phi_new)
        res = norm_phi_chg/norm_phi_new

        if ((res<tol) and (iter>5)):
            print("Convergence tolerance reached after %d iterations." %iter)
            break
        else:
            phi_old = phi_new
            phi_new = np.zeros((N_e))

    return x,phi_new


def Ana1(N_a,N_e):
    a = 0.0
    b = 1.0
    sigma_t = 1.0
    h = (b-a) / N_e
    mu, w = lg.LegendreRoots(N_a)

    psi_Left = np.zeros((N_a))
    psi_Right = np.zeros((N_a))
    phi_old = np.zeros((N_e))
    phi_new = np.zeros((N_e))

    A = 1.0/2/sigma_t


    x = np.zeros((N_e))
    for i in range(0, N_e):
        x[i] = a + 0.5 * h + i * h

    for i in range(0,N_e):
        for n in range(0,N_a):
            B = -1*sigma_t/mu[n]
            if (mu[n]>0):
                phi_new[i] = phi_new[i]+w[n]*A*(1-math.exp(B*(x[i])))
            else:
                phi_new[i] = phi_new[i] + w[n] * A * (1 - math.exp(B * (x[i]-b)))

    print("Sum of weights=%g" %(np.sum(w)))
    return x,phi_new

# N_b1=40
N_b2=100
# x3,tally = mc.MonteCarlo1G1D(0.0,1.0,200000,N_b1)



plt.figure(1)
plt.clf()


x1,phi1 = DD(64,50)
x2,phi2 = Ana1(64,N_b2)

plt.plot(x1,phi1,'kd',label="Diamond Difference")
plt.plot(x2,phi2,label="Analytical Solution")

plt.xlabel('X [cm]')
plt.ylabel(r'Scalar flux $\phi_0$')
plt.legend()
plt.savefig("OneDNoScattering.png")
plt.show()




print("End of program")