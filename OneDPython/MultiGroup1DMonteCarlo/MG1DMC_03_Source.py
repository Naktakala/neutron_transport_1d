import numpy as np
import math

X = 0
Y = 1
Z = 2

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class Particle:
  def __init__(self,Egrp,omega,pos=np.zeros(3),index=-1):
    self.Egrp = Egrp
    self.omega = omega
    self.pos = pos
    self.weight = 1.0
    self.index = index

    self.alive=True
    self.cur_matid=0

    self.rmc_sample_weight = 1.0



class MG1DMC_Methods:
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get source particle
  def GetSourceParticle(self,index):
    # ================================= Determine random angle direction
    theta = math.acos(self.rngen.RN()*2.0-1.0)
    vaphi = self.rngen.RN()*2.0*math.pi
    omega = np.zeros(3)
    omega[X] = math.sin(theta)*math.cos(vaphi)
    omega[Y] = math.sin(theta)*math.sin(vaphi)
    omega[Z] = math.cos(theta)

    newParticle = Particle(0,omega,np.array([0.0,0.0,-0.001]),index)

    if (omega[Z]>0.0):
      newParticle.weight = 1.0*omega[Z]

    # zmax = 0.125
    # #zmax = 0.001
    # newParticle=Particle(0, omega, np.array([0.0, 0.0, self.rngen.RN()*zmax]))
    # newParticle.weight = 1.0*zmax

    return newParticle

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Left surface
  def LeftIsotropicFlux(self,index, additional_w=1.0):
    # ================================= Determine random angle direction
    # mu = math.sqrt(self.rngen.RN())
    # theta = math.acos(mu)
    theta = math.acos(self.rngen.RN()*2.0-1.0)
    vaphi = self.rngen.RN()*2.0*math.pi
    omega = np.zeros(3)
    omega[X] = math.sin(theta)*math.cos(vaphi)
    omega[Y] = math.sin(theta)*math.sin(vaphi)
    omega[Z] = math.cos(theta)

    z = self.mesh.xmin-self.epsilon
    newParticle = Particle(0,omega,np.array([0.0,0.0,z]),index)

    newParticle.weight=1.0*math.cos(theta)*additional_w

    return newParticle

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Right surface
  def RightIsotropicFlux(self,index, additional_w=1.0):
    # ================================= Determine random angle direction
    theta = math.acos(self.rngen.RN()*2.0-1.0)
    vaphi = self.rngen.RN()*2.0*math.pi
    omega = np.zeros(3)
    omega[X] = math.sin(theta)*math.cos(vaphi)
    omega[Y] = math.sin(theta)*math.sin(vaphi)
    omega[Z] = math.cos(theta)

    z = self.mesh.xmax-self.epsilon
    newParticle = Particle(0,omega,np.array([0.0,0.0,z]),index)

    newParticle.weight=1.0*abs(math.cos(theta))*additional_w

    return newParticle

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Distributed source
  def DistributedIsotropic(self, index, additional_w=1.0):
    # ================================= Determine random angle direction
    theta=math.acos(self.rngen.RN()*2.0-1.0)
    vaphi=self.rngen.RN()*2.0*math.pi
    omega=np.zeros(3)
    omega[X]=math.sin(theta)*math.cos(vaphi)
    omega[Y]=math.sin(theta)*math.sin(vaphi)
    omega[Z]=math.cos(theta)

    z=self.mesh.xmin + self.rngen.RN()*(self.mesh.xmax - self.mesh.xmin)
    newParticle=Particle(0, omega, np.array([0.0, 0.0, z]), index)

    newParticle.weight=10.0*additional_w

    return newParticle

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Combine source
  def CombinationA(self,index):
    xi = self.rngen.RN()
    newParticle = []
    if (xi<0.33333333):
      newParticle = self.LeftIsotropicFlux(index,2.5/0.333333)
    elif (xi<0.666666):
      newParticle = self.RightIsotropicFlux(index,2.5/0.333333)
    else:
      newParticle = self.DistributedIsotropic(index,1/0.333333)

    return newParticle

  # ========================================== Monte-Carlo Solve
  def CustomSource(self, index):
    # ================================= Determine random angle direction
    theta=math.acos(self.rngen.RN()*2.0-1.0)
    vaphi=self.rngen.RN()*2.0*math.pi
    omega=np.zeros(3)
    omega[X]=math.sin(theta)*math.cos(vaphi)
    omega[Y]=math.sin(theta)*math.sin(vaphi)
    omega[Z]=math.cos(theta)

    z=self.mesh.xmin+self.rngen.RN()*5.0
    newParticle=Particle(0, omega, np.array([0.0, 0.0, z]), index)

    newParticle.weight=5.0

    return newParticle


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RMC source
  def RMCSource(self,index):

    elem_k=self.RMC_cur_elem
    cell_R_tot = abs(elem_k.residual_int_g[0]) + \
                 abs(elem_k.residual_s_0_g[0]) + \
                 abs(elem_k.residual_s_1_g[0])
    cell_weight = cell_R_tot/self.RMC_res_tot

    # ============================ Determine location
    sampleLeft = False
    sampleRite = False
    sampleCent = False
    rn = self.rngen.RN()
    if (rn<(abs(elem_k.residual_int_g[0])/cell_R_tot)):
      sampleCent=True
    else:
      rn=self.rngen.RN()
      if (rn<1.0/2.0):
        sampleLeft = True
      else:
        sampleRite = True


    # ============================ Sample left
    if sampleLeft:
      theta=math.acos(self.rngen.RN()*1.0)
      vaphi=self.rngen.RN()*2.0*math.pi
      omega=np.zeros(3)
      omega[X]=math.sin(theta)*math.cos(vaphi)
      omega[Y]=math.sin(theta)*math.sin(vaphi)
      omega[Z]=math.cos(theta)

      z=self.RMC_cur_elem.xi__+self.epsilon
      newParticle=Particle(0, omega, np.array([0.0, 0.0, z]), index)

      newParticle.weight=-cell_weight* \
                         self.RMC_cur_elem.residual_s_0_g[0]/abs(self.RMC_cur_elem.residual_s_0_g[0])* \
                         math.cos(theta)/2

    # ============================ Sample Rite
    if sampleRite:
      theta=math.acos(self.rngen.RN()*-1.0)
      vaphi=self.rngen.RN()*2.0*math.pi
      omega=np.zeros(3)
      omega[X]=math.sin(theta)*math.cos(vaphi)
      omega[Y]=math.sin(theta)*math.sin(vaphi)
      omega[Z]=math.cos(theta)

      z=self.RMC_cur_elem.xip1 - self.epsilon
      newParticle=Particle(0, omega, np.array([0.0, 0.0, z]), index)

      newParticle.weight=cell_weight* \
                         self.RMC_cur_elem.residual_s_1_g[0]/abs(self.RMC_cur_elem.residual_s_1_g[0])* \
                         math.cos(theta)/2

    # ============================ Sample Center
    if sampleCent:
      theta=math.acos(self.rngen.RN()*2.0-1.0)
      vaphi=self.rngen.RN()*2.0*math.pi
      omega=np.zeros(3)
      omega[X]=math.sin(theta)*math.cos(vaphi)
      omega[Y]=math.sin(theta)*math.sin(vaphi)
      omega[Z]=math.cos(theta)

      z=self.RMC_cur_elem.xi__ + self.rngen.RN()*self.RMC_cur_elem.h
      newParticle=Particle(0, omega, np.array([0.0, 0.0, z]), index)

      newParticle.weight=cell_weight* \
                         self.RMC_cur_elem.residual_int_g[0]/abs(self.RMC_cur_elem.residual_int_g[0])


    #newParticle.rmc_sample_weight = self.RMC_cur_elem.h

    return newParticle