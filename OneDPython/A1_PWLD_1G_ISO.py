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



#pdt_file_name =   "xs_PDTallSab_06000-c_146.data"
pdt_file_name = "xs_1_170.data"

Nel=240
mesh = MeshLib.OneD_Mesh(0,30,Nel)  #Mesh
L = 0                               #Scat order
N_a = 128                           #Number of angles

# ========================================== Define group structure
group_struct = GroupStructsLib.MakeSimple(1)
G = np.size(group_struct)           #Number of groups
#G=1
print("Number of groups read from data file: %d" %G)


# ========================================== Define materials
materials = []

m_testmaterial = NuclearMaterialsLib.NuclearMaterial1GIsotropic("Test1G",0.2,0.19)
materials.append(m_testmaterial)

#=========================================== Define BCs
bcs = []
bcs.append(np.array([PWLD.ISOTROPIC,PWLD.VACUUM])) #Boundary type ident
#bcs.append(np.array([PWLD.VACUUM,PWLD.VACUUM])) #Boundary type ident
bcs.append(np.zeros(G))                            #Left boundary groupwise values
bcs.append(np.zeros(G))                            #Right boundary groupwise values

bcs[1][0] = 1.0
source = np.zeros(Nel)+0.0
#source[0] = 1.0



#=========================================== Define Groupsets
groupsets = []
groupsets.append(np.array([0, 0, PWLD.WITH_DSA, 20, 150]))


#=========================================== Transport solve
solver = PWLD.PWLD(L,G,N_a,mesh,materials,source,bcs)
solver.Initialize()
#solver.SetOutputFileName("ZOut_240_8_3.txt")
#solver.ReadRestartData("ZOut_160_8_3.txt")
#solver.ReadRestartData("ZOut.txt")
solver.Solve(groupsets,group_struct)

x, phi = solver.GetSnPhi_g(0)

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



plt.figure(0)
plt.clf()
plt.plot(x, phi)

phi2 = np.copy(phi)

n = np.size(x)
for i in range(0,n):
  F = AnaSol.AngularFluxSlabPureAbsorber
  phi2[i] = RiemannIntegrate(F,0.0,math.pi*0.5,x[i])
  print("%g %g"%(x[i], phi[i]))

plt.plot(x, phi2)
plt.xlim([0,30])
# plt.ylim([0,1])
plt.show()


