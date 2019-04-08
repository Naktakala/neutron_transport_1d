import numpy as np
import math
import random

def MonteCarlo1G1D(a,b,N_p,N_b):
    dx = (b - a) / N_b
    xbins = np.zeros(N_b+1)
    for i in range(0,N_b+1):
        xbins[i] =  i*dx
    print(xbins)
    x = np.zeros((N_b))

    for i in range(0,N_b):
        x[i] = a + 0.5*dx + i*dx
    xtal  = np.zeros((N_b))
    sigma_t=1.0
    sigma_s=0.9*0
    last=0
    for p in range(0,N_p):
        if (p>=(last+100000)):
            print("Number of Particles:%d" %p)
            last = p
        #============================= Sample location
        #xpos = np.random.rand()*(b-a)+a
        xpos = random.uniform(0.0,1.0) * (b - a) + a
        mu = SampleAngle()
        #print("Source %g %g" %(xpos,mu))

        alive = True
        while alive:
            xposf,muf,alive = Transport(xpos,mu,sigma_t,sigma_s)
            xcon = AddToTally(xbins,N_b,xpos,mu,xposf)
            xtal = xtal + xcon
            xpos = xposf
            mu   = muf
            #print("  Reaction %g %g" % (xpos, mu))

    xtal = xtal/N_p

    return x,xtal

def Getxbin(x,xbins,N_b):
    binf = -1;
    for i in range(0, N_b):
        if ((x >= xbins[i]) and (x <= xbins[i + 1])):
            binf = i
            break

    if (binf<0):
        print("Error bin %g" %x)

    return binf


def SampleAngle():
    #mu = math.cos(np.random.rand() * math.pi)
    mu = math.cos(random.uniform(0.0, 1.0) * math.pi)

    return mu


def Transport(xpos,mui,sigma_t,sigma_s):
    # ==================== Distance to interaction
    rn = np.random.rand()
    rn = random.uniform(0.0, 1.0)
    d = -math.log(rn) / sigma_t

    # ==================== Distance to boundary
    dbx=0.0
    if (mui<0):
        dbx = (0-xpos)/mui
    else:
        dbx = (1-xpos)/mui

    #print("d=%g  dbx=%g" %(d,dbx))

    # ==================== Next position
    posf=xpos
    muf = mui
    alive = False
    if (d<dbx):
        posf = xpos + d*mui

        rn = np.random.rand()
        rn = random.uniform(0.0, 1.0)
        if (rn <= (sigma_s / sigma_t)):
            muf = SampleAngle()
            alive = True

    else:
        posf = xpos + dbx*mui

    if (posf<0.0):
        posf = 0.0;

    if (posf>1.0):
        posf = 1.0

    return posf,muf,alive

def AddToTally(xbins,N_b,xposi,mui,xposf):
    xcontrib = np.zeros((N_b))

    bini = Getxbin(xposi, xbins, N_b)
    binf = Getxbin(xposf, xbins, N_b)

    h = 1.0/N_b

    if (bini == binf):
        xcontrib[bini] = abs(xposf-xposi)

    elif (bini < binf):
        for i in range(bini,binf+1):
            if (i==bini):
                xcontrib[i] = abs(xbins[i+1]-xposi)
            elif (i==binf):
                xcontrib[i] = abs(xbins[i]-xposf)
            else:
                xcontrib[i] = abs(xbins[i+1]-xbins[i])

    elif (binf < bini):
        for i in range(binf,bini+1):
            if (i==bini):
                xcontrib[i] = abs(xbins[i]-xposi)
            elif (i==binf):
                xcontrib[i] = abs(xbins[i+1]-xposf)
            else:
                xcontrib[i] = abs(xbins[i+1]-xbins[i])


    return xcontrib/abs(mui)/h