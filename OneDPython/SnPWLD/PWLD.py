import numpy as np
import Legendre as lg
import time
import sys
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

import PWLD_1_Initialize
import AnalyticalSolutions

TYPEDEF = 0
GROUPVAL_LEFT = 1
GROUPVAL_RIGHT = 2

VACUUM = 0
ISOTROPIC = 1
REFLECTIVE = 2
LEFT = 0
RIGHT = 1

NO_DSA = 0
WITH_DSA = 1

# One dimensional Sn Transport solver
class PWLD(PWLD_1_Initialize.PWLD_Methods):
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constructor
  def __init__(self,L,G,Na,mesh,materials,source,bcs):
    self.L = L
    self.G = G
    self.Na = Na
    self.mesh = mesh
    self.materials = materials
    self.source = source
    self.bcs = bcs

    self.solve_tol = 1.0e-6

    self.P_ell_mu_n = np.zeros((L+1,Na))
    self.A = np.zeros((2,2))
    self.Ainv = np.zeros((2,2))
    self.b = np.zeros(2)
    self.psi = np.zeros(2)

    self.phig_dsa_new = np.zeros((self.G,self.mesh.Ndiv * 2))
    self.phig_dsa_old = self.phig_dsa_new.copy()
    self.A_dsa = np.zeros((self.mesh.Ndiv * 2, self.mesh.Ndiv * 2))
    self.b_dsa = np.zeros(self.mesh.Ndiv * 2)
    self.dsa_map = np.zeros(self.mesh.Ndiv * 2,dtype=int)

    self.AnalyticalSolution_Flag = False
    self.AnalyticalSolution = []
    self.AnalyticalSolution_x = []

    #============================= Initialize sweep orderings
    self.sweeporderings = []

    leftorder  = np.zeros(self.mesh.Ndiv,dtype=int)
    rightorder = np.zeros(self.mesh.Ndiv,dtype=int)

    for k in range(0, self.mesh.Ndiv):
      leftorder[k] = k
      rightorder[k] = self.mesh.Ndiv-k - 1

    self.sweeporderings.append(leftorder)
    self.sweeporderings.append(rightorder)

    self.outputFileName = ""
    self.outputFileNameSet = False


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Call main solve routine
  # This is the main control function
  def Solve(self,groupsets,group_struct):
    # =========================================== Loop over group sets
    num_grpsets=groupsets.__len__()
    print("Number of group sets: %d"%num_grpsets)
    tol=1.0e-6
    for gs in range(0, num_grpsets):
      gi=groupsets[gs][0]
      gf=groupsets[gs][1]
      plot_intv=groupsets[gs][3]

      res_gs = self.ComputePhiRelChange(gi,gf)/(gf-gi+1)
      print("Starting rel chng = %g" %res_gs)

      plot_counter=0
      stop_DSA=False
      prev_res=1.0
      for iter in range(0, groupsets[gs][4]):
        ti=time.time()
        relchg=self.TransportSweep(gi, gf, False, False)
        resid,maxres = self.ComputeResidual(gi, gf)
        spec_rad=relchg/prev_res
        prev_res=relchg
        tf=time.time()

        print("Grp %d-%d Iteration %d, Rel.Chg=%g, "
              "%g s, spec=%g, MaxRes=%g"%(gi, gf, iter, relchg, (tf-ti), spec_rad,maxres))

        if (groupsets[gs][2]==WITH_DSA) and (not stop_DSA):
          self.WGDSA_Solve(gi, gf, 30, 0)
          if (gf-gi)>0:
            self.TGDSA_Solve(gi, gf)

          if (self.AnalyticalSolution_Flag):
            E, phi=self.GetTotalSpectrum(group_struct)

            plt.figure(0)
            plt.clf()
            plt.loglog(E, phi)
            plt.loglog(self.AnalyticalSolution_x, self.AnalyticalSolution)
            plt.savefig("SnSpectrum.png")
            plt.show()

        if (relchg<self.solve_tol):
          break

        plot_counter+=1
        if plot_counter==plot_intv:
          if (self.AnalyticalSolution_Flag):
            E, phi=self.GetTotalSpectrum(group_struct)

            plt.figure(0)
            plt.clf()
            plt.loglog(E, phi)
            plt.loglog(self.AnalyticalSolution_x, self.AnalyticalSolution)
            plt.savefig("SnSpectrum.png")
            plt.show()

          plot_counter=0

      self.WriteRestartData()

