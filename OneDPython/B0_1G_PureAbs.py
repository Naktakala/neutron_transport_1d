import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
import sys

sys.path.append('./SnPWLD/')
sys.path.append('./NuclearMaterials/')
sys.path.append('./MultiGroup1DMonteCarlo/')
sys.path.append('./MeshLib/')

import MeshLib
import NuclearMaterialsLib
import GroupStructsLib
import PWLD
import AnalyticalSolutions as AnaSol
import MeshLib
import MG1DMC



Nel=40
mesh = MeshLib.OneD_Mesh(0,10,Nel)  #Mesh
mcmesh = MeshLib.OneD_Mesh(0,10,40) #Mesh for Monte-Carlo tallies
L = 0                               #Scat order
N_a = 32                           #Number of angles

# ========================================== Define group structure
group_struct = GroupStructsLib.MakeSimple(1)
G = np.size(group_struct)           #Number of groups
print("Number of groups read from data file: %d" %G)


# ========================================== Define materials
materials = []

m_testmaterial = NuclearMaterialsLib.NuclearMaterial1GIsotropic("Test1G",0.4,0.2)
materials.append(m_testmaterial)

#=========================================== Define BCs
bcs = []
bcs.append(np.array([PWLD.VACUUM,PWLD.VACUUM])) #Boundary type ident
bcs.append(np.zeros(G))                            #Left boundary groupwise values
bcs.append(np.zeros(G))                            #Right boundary groupwise values

# bcs[1][0] = 2.5/2.5
# bcs[2][0] = 2.5
source = np.zeros((G,Nel))+1.0

#=========================================== Define Groupsets
groupsets = []
groupsets.append(np.array([0, 0, PWLD.WITH_DSA, 20, 150]))

#=========================================== Transport solve
solver = PWLD.PWLD(L,G,N_a,mesh,materials,source,bcs)
solver.Initialize()
solver.SetOutputFileName("ZOut_B0.txt")
#solver.ReadRestartData("ZOut.txt")
solver.Solve(groupsets,group_struct)

# ========================================== Monte-Carlo Solve
num_particles = int(36000*6)
batch_size    = int(12000)
num_threads = 6
mcsource = []
mcsolver = MG1DMC.MultiGrp1DMC(L,G,mcmesh,materials,mcsource,bcs,group_struct,num_threads)
mcsolver.SourceRoutine = mcsolver.DistributedIsotropic  #LeftIsotropicFlux
mcsolver.SetOutputFileName("ZOut_B0_MC.txt")
#mcsolver.ReadRestartData("ZMCOut.txt")
mcsolver.RunMTForward(num_particles,batch_size,False)


# ========================================== Integration function for
#                                            analytical solution
def RiemannIntegrate(F,a,b,x,N=2000,angular=True):
  intgl = 0.0
  dtheta = (b-a)/N

  for i in range(0,N):
    theta = a+0.5*dtheta + dtheta*i
    f = F(x,theta)
    if angular:
      intgl += f*math.sin(theta)*dtheta
    else:
      intgl += f*dtheta

  return intgl

# ===================================== Print solutions
plt.figure(0)
plt.clf()

# ===================================== Print Sn-solution
print("Sn Solution:")
x, phi = solver.GetSnPhi_g(0)
plt.plot(x, phi, label='Sn Solution')
n = np.size(x)
for i in range(0,n):
  print("%g %g"%(x[i], phi[i]))

# ===================================== Print Monte-carlo solution
print("MC Solution")
x2, phi2, err2 = mcsolver.GetPhi_g(0)
plt.plot(x2,phi2,'kx',label='Monte-Carlo')
n = np.size(x2)
for i in range(0,n):
  print("%g %g (%.4f)"%(x2[i], phi2[i], err2[i]/phi2[i]))

# ===================================== Print Analytical solution
print("Analytical Solution:")
phi3 = np.copy(phi2)

n = np.size(x2)
for i in range(0,n):
  F = AnaSol.AngularFluxSlabPureAbsorber
  phi3[i] = RiemannIntegrate(F,0.0,math.pi*0.5,x2[i])
  #print("%g %g"%(x2[i], phi3[i]))

plt.plot(x2, phi3,label='Analytical Solution')
plt.xlim([0,10])
# plt.ylim([0,1])
plt.xlabel('Distance [cm]')
plt.ylabel('$\phi$',rotation=0)
plt.legend()
plt.savefig("Test_3_1.png")
plt.show()



