import numpy as np
import math

import MG1DMC_03_Source

X = 0
Y = 1
Z = 2

class MG1DMC_Methods(MG1DMC_03_Source.MG1DMC_Methods):
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Angular scattering
  def ScatterAngle(self,omega_i,mu):
    omega_f     = np.zeros(3)
    omega_fstar = np.zeros(3)

    # ================================= Scattering in reference frame
    if (mu>1.0): mu=1.0
    if (mu<-1.0): mu=-1.0
    mu = self.rngen.RN()*2.0-1.0
    theta = math.acos(mu)
    vaphi=self.rngen.RN()*2.0*math.pi
    omega_fstar[X]=math.sin(theta)*math.cos(vaphi)
    omega_fstar[Y]=math.sin(theta)*math.sin(vaphi)
    omega_fstar[Z]=math.cos(theta)

    # ================================= Rotation matrix
    khat=np.array([0.0, 0.0, 1.0])
    tangent=np.cross(khat, omega_i)
    tangent/=np.linalg.norm(tangent)
    binorm=np.cross(tangent, omega_i)
    binorm/=np.linalg.norm(binorm)

    R=np.zeros((3, 3))
    R[:, 0]=binorm[:]
    R[:, 1]=tangent[:]
    R[:, 2]=omega_i[:]

    omega_f=np.matmul(R, omega_fstar)

    omega_f/=np.linalg.norm(omega_f)

    after_mu = np.dot(omega_i,omega_f)

    #self.avg_mu += np.dot(omega_i,omega_f)
    self.cumul_mu += mu
    self.mu_calls += 1
    #print(self.avg_mu)

    return omega_f


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Raytracing
  def RayTrace(self,in_particle,id=0):
    # ====================================== Compute distance to interaction
    mat = self.materials[in_particle.cur_matid]
    d_to_interact = 0.0
    if (mat.sigma_t[in_particle.Egrp]>0.0):
      d_to_interact = -math.log(1.0-self.rngen.RN())/ \
                      mat.sigma_t[in_particle.Egrp]

    # ====================================== Compute distance to surface
    d_to_surface_up =  1.0e9
    d_to_surface_lo = -1.0e9
    if (abs(in_particle.omega[Z])>self.epsilon):
      d_to_surface_up = (self.mesh.xmax - in_particle.pos[Z])/ \
                        in_particle.omega[Z]
      d_to_surface_lo =(self.mesh.xmin - in_particle.pos[Z])/ \
                        in_particle.omega[Z]

    d_to_surface = 0.0
    if (d_to_surface_up>0.0):
      d_to_surface = d_to_surface_up
    else:
      d_to_surface = d_to_surface_lo

    # ====================================== Transport particle
    pos_f = in_particle.pos
    Eg_f = in_particle.Egrp
    omega_f = in_particle.omega
    if d_to_interact<d_to_surface:
      rn = self.rngen.RN()
      if (mat.sigma_t[Eg_f]>0.0):
        if (rn<=(mat.sigma_s[Eg_f]/(mat.sigma_t[Eg_f]))):
          pos_f = in_particle.pos + d_to_interact*in_particle.omega
          Eg_f = mat.SampleScatteringEnergyGrp(in_particle.Egrp,self.rngen.RN())
          mu = mat.SampleScatteringAngle(in_particle.Egrp,Eg_f,self.rngen.RN())
          omega_f = self.ScatterAngle(in_particle.omega,mu)
        else:
          pos_f=in_particle.pos+d_to_interact*in_particle.omega
          in_particle.alive = False
      else:
        pos_f = in_particle.pos + d_to_surface*in_particle.omega
        in_particle.alive=False
    else:
      pos_f = in_particle.pos + d_to_surface*in_particle.omega
      in_particle.alive = False


    # ====================================== Contribute to tallies
    self.ContribTallies(in_particle,pos_f)

    in_particle.pos = pos_f
    in_particle.Egrp = Eg_f
    in_particle.omega = omega_f

    return in_particle


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Contribute to tallies
  def ContribTallies(self,particle,posf):
    glob = self.glob_tally
    elem = self.elem_tally

    Egrp = particle.Egrp
    posi = particle.pos
    weight = particle.weight
    orig_posi = particle.pos
    orig_posf = posf

    # if (posf[Z]<posi[Z]):
    #   temp = posf
    #   posf = posi
    #   posi = temp
    #
    dz = abs(posf[Z]-posi[Z])

    # ====================================== Global spectrum tally
    tracklength = np.linalg.norm(posf-posi)

    glob[Egrp]+=tracklength/(self.mesh.xmax-self.mesh.xmin)

    # ====================================== Element tallies
    for k_outer in range(0,self.mesh.Ndiv):
      k = k_outer
      if (particle.omega[Z]<=0.0):
        k = self.mesh.Ndiv - k_outer - 1
      elem_k = self.mesh.elements[k]

      contrib = 0.0
      crossing_cell = False

      if (posf[Z]>=posi[Z]):
        if (posf[Z]>= elem_k.xip1) and (posi[Z]<= elem_k.xi__):
          contrib += tracklength*(elem_k.h/dz)/elem_k.h
          crossing_cell = True
        elif (posf[Z]<= elem_k.xip1) and (posf[Z]>= elem_k.xi__) and (posi[Z]<= elem_k.xi__):
          contrib += tracklength*(posf[Z] - elem_k.xi__)/dz/elem_k.h
        elif (posf[Z]>= elem_k.xip1) and (posi[Z]>= elem_k.xi__) and (posi[Z]<= elem_k.xip1):
          contrib += tracklength*(elem_k.xip1 - posi[Z])/dz/elem_k.h
          crossing_cell = True
        elif (posf[Z]<= elem_k.xip1) and (posf[Z]>= elem_k.xi__) and \
          (posi[Z]<= elem_k.xip1) and (posi[Z]>= elem_k.xi__):
          contrib += tracklength/elem_k.h
      else:
        if (posi[Z]>= elem_k.xip1) and (posf[Z]<= elem_k.xi__):
          contrib += tracklength*(elem_k.h/dz)/elem_k.h
          crossing_cell = True
        elif (posi[Z]<= elem_k.xip1) and (posi[Z]>= elem_k.xi__) and (posf[Z]<= elem_k.xi__):
          contrib += tracklength*(posi[Z] - elem_k.xi__)/dz/elem_k.h
        elif (posi[Z]>= elem_k.xip1) and (posf[Z]>= elem_k.xi__) and (posf[Z]<= elem_k.xip1):
          contrib += tracklength*(elem_k.xip1 - posf[Z])/dz/elem_k.h
          crossing_cell = True
        elif (posi[Z]<= elem_k.xip1) and (posi[Z]>= elem_k.xi__) and \
          (posf[Z]<= elem_k.xip1) and (posf[Z]>= elem_k.xi__):
          contrib += tracklength/elem_k.h

      weight=particle.weight*particle.rmc_sample_weight
      elem[Egrp, k] += contrib*weight
      self.elem_tally2[Egrp][k].Contrib(contrib*weight,particle.index)

      if ((self.RMC_active) and (crossing_cell)):
        if (particle.omega[Z]>0.0):
          if (k<(self.RMC_mesh.Ndiv-1)):
            particle.weight += self.RMC_jumps[Egrp,k]
        else:
          if (k>0):
            particle.weight -= self.RMC_jumps[Egrp,k-1]

    posi = orig_posi
    posf = orig_posf