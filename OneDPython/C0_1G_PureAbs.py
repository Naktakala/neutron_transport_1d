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
mcmesh = MeshLib.OneD_Mesh(0,10,Nel) #Mesh for Monte-Carlo tallies
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
bcs.append(np.array([PWLD.ISOTROPIC,PWLD.ISOTROPIC])) #Boundary type ident
bcs.append(np.zeros(G))                            #Left boundary groupwise values
bcs.append(np.zeros(G))                            #Right boundary groupwise values

bcs[1][0] = 1.0
# bcs[2][0] = 2.5
source = np.zeros((G,Nel))+0.0

# for k in range(0,20):
#   source[0,k] = 1.0

#=========================================== Define Groupsets
groupsets = []
groupsets.append(np.array([0, 0, PWLD.WITH_DSA, 20, 150]))

#=========================================== Transport solve
solver = PWLD.PWLD(L,G,N_a,mesh,materials,source,bcs)
solver.Initialize()
solver.SetOutputFileName("ZOut_B0.txt")
#solver.ReadRestartData("ZOut.txt")
solver.Solve(groupsets,group_struct)

for k in range(0,mesh.Ndiv):
  elem_k = mesh.elements[k]
  # elem_k.phi_new_mg_0[0][0] *= 0.7
  # elem_k.phi_new_mg_1[0][0] *= 0.7
  # elem_k.phi_new_mg_avg[0][0] *= 0.7

  # elem_k.phi_new_mg_0[0][0]*=1.3
  # elem_k.phi_new_mg_1[0][0]*=1.3
  # elem_k.phi_new_mg_avg[0][0]*=1.3

solver.ComputeResidual(0,0)
solver.WriteRestartData()


#=========================================== MonteCarlo solve
num_particles = int(36000*60)
batch_size    = int(36000)
num_threads = 6
mcsource = []
mcsolver = MG1DMC.MultiGrp1DMC(L,G,mcmesh,materials,mcsource,bcs,group_struct,num_threads)
mcsolver.SourceRoutine = mcsolver.LeftIsotropicFlux #DistributedIsotropic #LeftIsotropicFlux #mcsolver.CustomSource
mcsolver.RunMTForward(num_particles,batch_size,False)








# ===================================== Print solutions
plt.figure(0)
plt.clf()

# ===================================== Print Sn-solution
print("Sn Solution:")
x, phi = solver.GetAvgSnPhi_g(0)
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
  print("%3d %g %g (%.4f)"%(i,x2[i], phi2[i], err2[i]/phi2[i]))




plt.xlim([0,10])
# plt.ylim([0,1])
plt.xlabel('Distance [cm]')
plt.ylabel('$\phi$',rotation=0)
plt.legend()
# plt.savefig("Test_3_1.png")
plt.show()


# ========================================== RMC solve
num_particles = int(12000)
batch_size    = int(12000)
num_threads = 6
mcsource = []
rmcsolver = MG1DMC.MultiGrp1DMC(L,G,mcmesh,materials,mcsource,bcs,group_struct,num_threads)
err_rmc = np.zeros(mesh.Ndiv)
for k in range(0,mesh.Ndiv):
  print("========= Element %d =========" %k)
  rmcsolver.InitializeRMC(mesh,k,0)
  rmcsolver.RunMTForward(num_particles, batch_size, False)
  xe,et,std = rmcsolver.GetPhi_g(0)
  err_rmc += et
  rmcsolver.ZeroOutTallies()
err_rmc = err_rmc*10/mesh.Ndiv

# ==================================================== Plot error
plt.figure(1)
plt.clf()

err1 = phi.copy()
n = np.size(x)
for i in range(0,n):
  err1[i] = phi[i]-phi2[i]

plt.plot(x,err1,'k-',label='True error')
plt.plot(x,err_rmc,'kx',label='RMC error')
plt.xlim([0,10])
# plt.ylim([0,1])
plt.xlabel('Distance [cm]')
plt.ylabel('$\phi$',rotation=0)
plt.legend()
plt.show()

print("Total true error %g" %np.sum(err1))
print("Total RMC error %g" %np.sum(err_rmc))
print("Ratio RMC/True = %g" %(np.sum(err_rmc)/np.sum(err1)))
