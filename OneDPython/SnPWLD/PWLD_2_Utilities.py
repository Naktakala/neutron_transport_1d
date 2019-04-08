import numpy as np
import PWLD_3_TransportSweep
import sys

class PWLD_Methods(PWLD_3_TransportSweep.PWLD_Methods):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Print averaged flux
    def PrintPhi(self, ell=0):
      for k in range(0, self.mesh.Ndiv):
        for g in range(0, self.G):
          avg_phi=0.5*self.mesh.elements[k].phi_old_mg_0[ell][g]+ \
                  0.5*self.mesh.elements[k].phi_old_mg_1[ell][g]

          print("Element %3d, moment %2d, grp %3d, phi_avg= %g"%(k, ell, g, avg_phi))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Print average flux for a group
    def PrintPhiGrp(self, g, ell=0):
      for k in range(0, self.mesh.Ndiv):
        avg_phi=0.5*self.mesh.elements[k].phi_old_mg_0[ell][g]+ \
                0.5*self.mesh.elements[k].phi_old_mg_1[ell][g]

        print("Element %3d, moment %2d, grp %3d, phi_avg= %g"%(k, ell, g, avg_phi))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get the total problem spectrum
    def GetTotalSpectrum(self, group_structure, div_by_bin=True):
      flux=np.zeros(self.G)
      Ecenter=np.zeros(self.G)

      for g in range(0, self.G):
        Ewidth=0.0
        if (g<(self.G-1)):
          Ecenter[g]=0.5*group_structure[g]+ \
                     0.5*group_structure[g+1]
          Ewidth=group_structure[g]- \
                 group_structure[g+1]
        else:
          Ecenter[g]=0.5*group_structure[g]
          Ewidth=group_structure[g]

        for k in range(0, self.mesh.Ndiv):
          elem_avg=0.5*self.mesh.elements[k].phi_new_mg_0[0][g]+ \
                   0.5*self.mesh.elements[k].phi_new_mg_1[0][g]

          flux[g]+=elem_avg/Ewidth/self.mesh.Ndiv

      return Ecenter/1.0e6, flux

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get discontinues flux for a group
    def GetSnPhi_g(self, g):
      x=np.zeros(self.mesh.Ndiv*2)
      phi_g=np.zeros(self.mesh.Ndiv*2)

      di=-2
      for k in range(0, self.mesh.Ndiv):
        di+=2
        elem_k=self.mesh.elements[k]
        x[di]=elem_k.xi__
        x[di+1]=elem_k.xip1
        phi_g[di]=elem_k.phi_new_mg_0[0][g]
        phi_g[di+1]=elem_k.phi_new_mg_1[0][g]

      return x, phi_g

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Get Average flux for a group
    def GetAvgSnPhi_g(self, g):
      x=np.zeros(self.mesh.Ndiv)
      phi_g=np.zeros(self.mesh.Ndiv)

      for k in range(0, self.mesh.Ndiv):
        elem_k = self.mesh.elements[k]
        x[k]     = elem_k.xi__*0.5 + elem_k.xip1*0.5
        phi_g[k] = elem_k.phi_new_mg_avg[0][g]

      return x, phi_g

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Set output file name
    def SetOutputFileName(self,file_name):
      self.outputFileName = file_name
      self.outputFileNameSet = True

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write restart data
    def WriteRestartData(self):
      file_name = "ZOut.txt"
      if self.outputFileNameSet:
        file_name = self.outputFileName
      print("Writing restart data to " + file_name)

      ofile = open(file_name,'w')

      ofile.write("NumElements %d\n" %self.mesh.Ndiv)
      ofile.write("NumGroups %d\n" %self.G)
      ofile.write("NumMoments %d\n" %self.L)


      for k in range(0,self.mesh.Ndiv):
        elem_k = self.mesh.elements[k]
        ofile.write("#============================\n")
        ofile.write("Element %4d\n" %k)

        for g in range(0, self.G):
          resid=elem_k.residual_tot_g[g]
          ofile.write("Residual %3d %.12e\n" %(g, resid))
          ofile.write("InteriorResidual %3d %.12e\n" %(g, elem_k.residual_int_g))
          ofile.write("LeftResidual     %3d %.12e\n" %(g, elem_k.residual_s_0_g))
          ofile.write("RightResidual    %3d %.12e\n" %(g, elem_k.residual_s_1_g))
          ofile.write("LeftNetIncoming  %3d %.12e\n" %(g,elem_k.total_psiL_incoming[g]))
          ofile.write("LeftNetOutgoing  %3d %.12e\n" %(g,elem_k.total_psiL_outgoing[g]))
          ofile.write("RightNetIncoming %3d %.12e\n" %(g,elem_k.total_psiR_incoming[g]))
          ofile.write("RightNetOutgoing %3d %.12e\n" %(g,elem_k.total_psiR_outgoing[g]))

        ofile.write("Vertex 0 z %g\n" %elem_k.xi__)
        for ell in range(0,self.L+1):
          ofile.write("Moment " + str(ell)+"\n")
          for g in range(0,self.G):
            oflux=elem_k.phi_old_mg_0[ell][g]
            ofile.write("Group %3d %.12e\n" %(g,oflux))

        ofile.write("Vertex 1 z %g\n"%elem_k.xip1)
        for ell in range(0, self.L+1):
          ofile.write("Moment "+str(ell)+"\n")
          for g in range(0, self.G):
            oflux=elem_k.phi_old_mg_1[ell][g]
            ofile.write("Group %3d %.12e\n"%(g, oflux))

      ofile.close()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Read restart data
    def ReadRestartData(self,file_name):
      ifile = open(file_name,'r')

      auxlines=ifile.readlines()
      ifile.seek(0, 0)

      in_num_el = 0
      in_num_grp = 0
      in_num_moms = 0
      outer_line_num=-1
      kel = -1
      vert = -1
      ell = -1
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

        if (linewords[0] == "NumMoments"):
          in_num_moms = int(linewords[1])
          if (in_num_moms > self.L):
            print("Error incorrect amount of moments in restart file")
            sys.exit(IOError)

        if (linewords[0] == "Element"):
          kel = int(linewords[1])

        if (linewords[0] == "Vertex"):
          vert = int(linewords[1])

          if vert==0:
            self.mesh.elements[kel].xi__ = float(linewords[3])
          else:
            self.mesh.elements[kel].xip1 = float(linewords[3])

        if (linewords[0] == "Moment"):
          ell = int(linewords[1])

        if (linewords[0] == "Group"):
          grp = int(linewords[1])

          if vert==0:
            self.mesh.elements[kel].phi_new_mg_0[ell][grp] = float(linewords[2])
          else:
            self.mesh.elements[kel].phi_new_mg_1[ell][grp]=float(linewords[2])

        if (linewords[0] == "Residual"):
          grp = int(linewords[1])
          self.mesh.elements[kel].residual_tot_g[grp] = float(linewords[2])


      for g in range(0, self.G):
        for k in range(0, self.mesh.Ndiv):
          for ell in range(0, self.L+1):
            self.mesh.elements[k].phi_old_mg_0[ell][g]= \
              self.mesh.elements[k].phi_new_mg_0[ell][g]
            self.mesh.elements[k].phi_old_mg_1[ell][g]= \
              self.mesh.elements[k].phi_new_mg_1[ell][g]


