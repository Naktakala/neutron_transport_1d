import numpy as np
import math
import sys

import MG1DMC_02_RayTrace

X = 0
Y = 1
Z = 2


class MG1DMC_Methods(MG1DMC_02_RayTrace.MG1DMC_Methods):
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Set output file name
  def SetOutputFileName(self, file_name):
    self.outputFileName=file_name
    self.outputFileNameSet=True

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get the total spectrum
  def GetTotalSpectrum(self,num_particles,div_by_bin=True):
    flux = np.zeros(self.G)
    Ecenter = np.zeros(self.G)

    group_structure = self.group_struct

    for g in range(0, self.G):
      Ewidth = 0.0
      if (g<(self.G-1)):
        Ecenter[g] = 0.5*group_structure[g]+ \
                     0.5 * group_structure[g+1]
        Ewidth = group_structure[g]- \
                 group_structure[g+1]
      else:
        Ecenter[g] = 0.5 * group_structure[g]
        Ewidth = group_structure[g]

      flux[g] += self.comb_glob_tally[g]/Ewidth

    return Ecenter/1.0e6,flux/num_particles



  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get a spatial vector for a group
  def GetPhi_g(self,g):
    x = np.zeros(self.mesh.Ndiv)
    phi = np.zeros(self.mesh.Ndiv)
    stddev=np.zeros(self.mesh.Ndiv)

    for k in range(0,self.mesh.Ndiv):
      elem_k = self.mesh.elements[k]
      x[k] = (elem_k.xi__ + elem_k.xip1)*0.5
      # phi[k] = self.comb_elem_tally[g,k]
      phi[k]=self.comb_elem_tally2[g][k].Mean()
      stddev[k] = self.comb_elem_tally2[g][k].StdDev()

    # return x, phi/self.particles_ran

    return x, phi, stddev

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get Residuals
  def GetResiduals(self,g):
    x=np.zeros(self.RMC_mesh.Ndiv)
    res=np.zeros(self.RMC_mesh.Ndiv)

    for k in range(0, self.RMC_mesh.Ndiv):
      elem_k=self.RMC_mesh.elements[k]
      x[k]=(elem_k.xi__+elem_k.xip1)*0.5
      res[k] = elem_k.residual_tot_g[g]

    return x, res

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write restart data
  def WriteRestartData(self):
    file_name="ZMCOut.txt"
    if self.outputFileNameSet:
      file_name=self.outputFileName
    #print("Writing restart data to "+file_name)

    ofile=open(file_name, 'w')

    ofile.write("NumElements %d\n"%self.mesh.Ndiv)
    ofile.write("NumGroups %d\n"%self.G)
    ofile.write("NumParticlesRan %d\n"%self.particles_ran)

    for g in range(0, self.G):
      spectrum_g = self.comb_glob_tally[g]
      spectrum_g_norm = spectrum_g/self.particles_ran
      ofile.write("Spectrum %3d %.14e %.14e\n" %(g,spectrum_g,spectrum_g_norm))

    for g in range(0, self.G):
      ofile.write("Group   %4d\n" %g)
      for k in range(0, self.mesh.Ndiv):
        elem_k = self.mesh.elements[k]
        oflux  = self.comb_elem_tally[g, k]
        oflux_norm = self.comb_elem_tally[g, k]/self.particles_ran
        ofile.write("Element %4d %.14e %.14e\n" %(k,oflux,oflux_norm))

    ofile.close()

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Read restart data
  def ReadRestartData(self,file_name):
    ifile = open(file_name,'r')

    auxlines=ifile.readlines()
    ifile.seek(0, 0)

    in_num_el = 0
    in_num_grp = 0
    outer_line_num=-1
    kel = -1
    grp = -1
    while (outer_line_num<(len(auxlines)-1)):
      outer_line_num+=1

      linewords=auxlines[outer_line_num].split()

      if (len(linewords)<2):
        continue

      if (linewords[0] == "NumElements"):
        in_num_el = int(linewords[1])
        if (in_num_el != self.mesh.Ndiv):
          print("Error incorrect amount of elements in restart file")
          sys.exit(IOError)

      if (linewords[0] == "NumGroups"):
        in_num_grp = int(linewords[1])
        if (in_num_grp != self.G):
          print("Error incorrect amount of groups in restart file")
          sys.exit(IOError)

      if (linewords[0] == "NumParticlesRan"):
        self.particles_ran = int(linewords[1])

      if (linewords[0] == "Spectrum"):
        grp = int(linewords[1])
        self.comb_glob_tally[grp] = float(linewords[2])

      if (linewords[0] == "Group"):
        grp = int(linewords[1])

      if (linewords[0] == "Element"):
        kel = int(linewords[1])
        self.comb_elem_tally[grp,kel] = float(linewords[2])

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize RMC
  def InitializeRMC(self,rmc_mesh,el_num,g):
    self.RMC_mesh = rmc_mesh
    self.RMC_cur_elem = self.RMC_mesh.elements[el_num]

    self.SourceRoutine = self.RMCSource

    if (not self.RMC_res_tot_set):
      self.RMC_res_tot_set = True

      for k in range(0,self.RMC_mesh.Ndiv):
        elem_k = self.RMC_mesh.elements[k]
        self.RMC_res_tot += abs(elem_k.residual_int_g[0]) + \
                            abs(elem_k.residual_s_0_g[0]) + \
                            abs(elem_k.residual_s_1_g[0])



    #self.RMC_active = True
    print("Done initializing")


