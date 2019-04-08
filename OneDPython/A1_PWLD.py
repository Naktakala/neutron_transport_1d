import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
import sys

sys.path.append('./SnPWLD/')
sys.path.append('./NuclearMaterials/')
sys.path.append('./MultiGroup1DMonteCarlo/')

import MeshLib
import NuclearMaterialsLib
import GroupStructsLib
import PWLD
import AnalyticalSolutions
import MG1DMC



#pdt_file_name =   "xs_PDTallSab_06000-c_146.data"
pdt_file_name = "xs_1_170.data"

Nel=240
mesh = MeshLib.OneD_Mesh(0,30,Nel)  #Mesh
L = 3                               #Scat order
N_a = 8                             #Number of angles

# ========================================== Define group structure
group_struct = GroupStructsLib.GetFromPDTdata(pdt_file_name)
G = np.size(group_struct)           #Number of groups
#G=1
print("Number of groups read from data file: %d" %G)

# ========================================== Define materials
materials = []
m_graphite = NuclearMaterialsLib.NuclearMaterial("Graphite", L, G)
#m_graphite.InitializeFromPDTdata(pdt_file_name,0.08475,"2519")
m_graphite.InitializeFromPDTdata(pdt_file_name)
m_graphite.InitializeScatteringTables()

materials.append(m_graphite)

# m_testmaterial = MaterialsLib.NuclearMaterial1GIsotropic("Test1G",1.0,0.0)
# materials.append(m_testmaterial)

#=========================================== Define BCs
bcs = []
bcs.append(np.array([PWLD.ISOTROPIC,PWLD.VACUUM])) #Boundary type ident
bcs.append(np.zeros(G))                            #Left boundary groupwise values
bcs.append(np.zeros(G))                            #Right boundary groupwise values

bcs[1][0] = 1.0

#=========================================== Define Groupsets
groupsets = []
groupsets.append(np.array([0, 40, PWLD.NO_DSA, 20, 150]))
groupsets.append(np.array([41, 51, PWLD.NO_DSA, 20, 150]))
groupsets.append(np.array([52, 62, PWLD.NO_DSA, 20, 150]))
groupsets.append(np.array([63, 167, PWLD.WITH_DSA, 20, 1000]))


#=========================================== Transport solve
solver = PWLD.PWLD(L,G,N_a,mesh,materials,np.zeros(Nel),bcs)
solver.Initialize()
#solver.SetOutputFileName("ZOut_240_8_3.txt")
#solver.ReadRestartData("ZOut_160_8_3.txt")
#solver.ReadRestartData("ZOut.txt")
solver.Solve(groupsets,group_struct)

E, phi = solver.GetTotalSpectrum(group_struct)

plt.figure(0)
plt.clf()
plt.loglog(E, phi)
plt.loglog(E, AnalyticalSolutions.MCNP1)

plt.show()


#=========================================== Plot spatial distribution
# plt.figure(0)
# plt.clf()
# x1,phi1 = solver.GetSnPhi_g(55)
# x2,phi2 = mcsolver.GetPhi_g(num_particles,55)
# plt.plot(x1,phi1)
# plt.plot(x2,phi2)
# plt.show()

# ========================================== Visualize Transfer matrix
# plt.figure(figsize=(6,6))
# Atest = np.matrix.transpose(m_graphite.sigma_s_mom[0])
# plt.imshow(np.log10(Atest)+10, cmap=cm.Greys_r)
# plt.xlabel('Destination energy group')
# plt.ylabel('Source energy group')
# plt.savefig("SERPENTTransferMatrix.png")
# plt.show()



# x,phi_mom = OneDPWLD(mesh,L,N_a
