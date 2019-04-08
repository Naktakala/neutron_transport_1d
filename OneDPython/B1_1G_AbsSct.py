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



Nel=80
mesh = MeshLib.OneD_Mesh(0,30,Nel)  #Mesh
mcmesh = MeshLib.OneD_Mesh(0,30,80) #Mesh for Monte-Carlo tallies
L = 0                               #Scat order
N_a = 32                           #Number of angles

# ========================================== Define group structure
group_struct = GroupStructsLib.MakeSimple(1)
G = np.size(group_struct)           #Number of groups
print("Number of groups read from data file: %d" %G)


# ========================================== Define materials
materials = []

m_testmaterial = NuclearMaterialsLib.NuclearMaterial1GIsotropic("Test1G",0.2,0.19)
materials.append(m_testmaterial)

#=========================================== Define BCs
bcs = []
bcs.append(np.array([PWLD.ISOTROPIC,PWLD.VACUUM])) #Boundary type ident
bcs.append(np.zeros(G))                            #Left boundary groupwise values
bcs.append(np.zeros(G))                            #Right boundary groupwise values

bcs[1][0] = 1.0
source = np.zeros((G,Nel))+0.0

#=========================================== Define Groupsets
groupsets = []
groupsets.append(np.array([0, 0, PWLD.WITH_DSA, 20, 150]))

#=========================================== Transport solve
solver = PWLD.PWLD(L,G,N_a,mesh,materials,source,bcs)
solver.Initialize()
solver.SetOutputFileName("ZOut_B1.txt")
#solver.ReadRestartData("ZOut.txt")
solver.Solve(groupsets,group_struct)

# ========================================== Monte-Carlo Solve
num_particles = int(288000)
batch_size    = int(24000)
num_threads = 6
mcsource = []
mcsolver = MG1DMC.MultiGrp1DMC(L,G,mcmesh,materials,mcsource,bcs,group_struct,num_threads)
mcsolver.SetOutputFileName("ZOut_B1_MC.txt")
#mcsolver.ReadRestartData("ZMCOut.txt")
mcsolver.RunMTForward(num_particles,batch_size,False)


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


plt.xlim([0,30])
# plt.ylim([0,1])
plt.xlabel('Distance [cm]')
plt.ylabel('$\phi$',rotation=0)
plt.legend()
plt.show()


