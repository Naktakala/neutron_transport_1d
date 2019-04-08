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
import MG1DMC
import MeshLib



#pdt_file_name =   "xs_PDTallSab_06000-c_146.data"
pdt_file_name = "xs_1_170.data"

Nel=80
mesh = MeshLib.OneD_Mesh(0,30,Nel)  #Mesh
L = 0                               #Scat order
N_a = 8                             #Number of angles

# ========================================== Define group structure
group_struct = GroupStructsLib.MakeSimple(1)
G = np.size(group_struct)           #Number of groups
#G=1
print("Number of groups read from data file: %d" %G)


# ========================================== Define materials
materials = []

m_testmaterial = NuclearMaterialsLib.NuclearMaterial1GIsotropic("Test1G",0.02,0.019)
materials.append(m_testmaterial)

#=========================================== Define BCs
bcs = []
bcs.append(np.array([PWLD.ISOTROPIC,PWLD.VACUUM])) #Boundary type ident
bcs.append(np.zeros(G))                            #Left boundary groupwise values
bcs.append(np.zeros(G))                            #Right boundary groupwise values

bcs[1][0] = 1.0



#=========================================== Define Groupsets
groupsets = []
groupsets.append(np.array([0, 0, PWLD.NO_DSA, 20, 150]))


#=========================================== Transport solve
#=========================================== MC solve
mcsolver = MG1DMC.MultiGrp1DMC(L,G,mesh,materials,np.zeros(Nel),bcs,group_struct,6)
num_particles = int(288000*8)
batch_size    = int(48000)
#mcsolver.ReadRestartData("ZMCOut.txt")
mcsolver.RunMTForward(num_particles,batch_size,False)

x, phi,stddev = mcsolver.GetPhi_g(0)


plt.figure(0)
plt.clf()
plt.plot(x, phi)

phi_top = phi.copy()+stddev
phi_bot = phi.copy()-stddev
plt.plot(x, phi_top)
plt.plot(x, phi_bot)


n = np.size(x)
for i in range(0,n):
  print("%g %g %g"%(x[i], phi[i], stddev[i]/phi[i]))


plt.xlim([0,30])
#plt.ylim([0,1.6])
plt.show()

print("Average scattering angle = %g" %(mcsolver.avg_mu/mcsolver.particles_ran))
print("Flux at x=10.0 : %g, should be %g" %(phi[26],1.076))


