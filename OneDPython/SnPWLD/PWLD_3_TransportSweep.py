import numpy as np
import Legendre as lg
import time
import math

import PWLD_4_WGDSA

TYPEDEF = 0
GROUPVAL_LEFT = 1
GROUPVAL_RIGHT = 2

VACUUM = 0
ISOTROPIC = 1
REFLECTIVE = 2
LEFT = 0
RIGHT = 1

class PWLD_Methods(PWLD_4_WGDSA.PWLD_Methods):
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  def TransportSweep(self, gs_i, gs_f, lumped=False, verbose=False):


    t_sweepout = time.time()
    t_sweepout1 = 0.0
    t_sweepout2 = 0.0
    t_sweepout3 = 0.0

    for g in range(gs_i, gs_f+1):
      for k in range(0, self.mesh.Ndiv):
        elem_k = self.mesh.elements[k]
        for ell in range(0, self.L+1):
          elem_k.phi_old_mg_0[ell][g] = elem_k.phi_new_mg_0[ell][g]
          elem_k.phi_old_mg_1[ell][g] = elem_k.phi_new_mg_1[ell][g]
        elem_k.total_psiL_outgoing[g] = 0.0
        elem_k.total_psiL_incoming[g] = 0.0
        elem_k.total_psiR_outgoing[g] = 0.0
        elem_k.total_psiR_incoming[g] = 0.0

    for g in range(gs_i, gs_f + 1):
      # ============================== Compute right hand side
      if verbose: print("Computing b")
      t_compb = time.time()
      for k in range(0, self.mesh.Ndiv):
        elem_k = self.mesh.elements[k]
        mat = self.materials[self.mesh.elements[k].mat_id]
        for ell in range(0, self.L + 1):

          sum_val_lumped = 0.0
          sum_val_unlumped_0 = 0.0
          sum_val_unlumped_1 = 0.0
          gi = mat.sigma_s_mom_limits[ell][g, 0]
          gf = mat.sigma_s_mom_limits[ell][g, 1]

          for gprime in range(gi, gf + 1):
            sigs = mat.sigma_s_mom[ell][gprime, g]

            if lumped:
                avg_phi = \
                    0.5 * elem_k.phi_old_mg_0[ell][gprime] + \
                    0.5 * elem_k.phi_old_mg_1[ell][gprime]
                sum_val_lumped += sigs * avg_phi
            else:
                sum_val_unlumped_0 += \
                    sigs * elem_k.phi_old_mg_0[ell][gprime]
                sum_val_unlumped_1 += \
                    sigs * elem_k.phi_old_mg_1[ell][gprime]

          elem_k.phi_new_mg_0[ell][g] = 0.0
          elem_k.phi_new_mg_1[ell][g] = 0.0

          if lumped:
            elem_k.scat_src_mg[ell, g] = \
                  0.5 * (2 * ell + 1) * sum_val_lumped
          else:
            elem_k.scat_src_mg_0[ell, g] = \
                  0.5 * (2 * ell + 1) * sum_val_unlumped_0
            elem_k.scat_src_mg_1[ell, g] = \
                  0.5 * (2 * ell + 1) * sum_val_unlumped_1
      t_compb = time.time() - t_compb

      if verbose: print("Sweeping:")
      for n in range(0, self.Na):
        mu = self.mu_n[n]

        sweepordering = []
        if (mu < 0.0):
          sweepordering = self.sweeporderings[RIGHT]
        else:
          sweepordering = self.sweeporderings[LEFT]

        # ================================================== Loop over cells
        for ks in range(0, self.mesh.Ndiv):
          k = sweepordering[ks]
          elem_k = self.mesh.elements[k]
          mat = self.materials[elem_k.mat_id]

          # ========================================= Loop over groups in Groupset

          t_sweepout1a = time.time()
          # ================================= Determine upwind flux
          psi_upwind = 0.0
          if mu < 0.0:
            if (k == (self.mesh.Ndiv - 1)) and (self.bcs[TYPEDEF][RIGHT] == ISOTROPIC):
              psi_upwind = self.bcs[GROUPVAL_RIGHT][g] / 2.0
            else:
              if (k < ((self.mesh.Ndiv - 1))):
                psi_upwind = self.mesh.elements[k + 1].psi_out[g]
          else:
            if (k == 0) and (self.bcs[TYPEDEF][LEFT] == ISOTROPIC):
              psi_upwind = self.bcs[GROUPVAL_LEFT][g] / 2.0
            else:
              if k > 0:
                psi_upwind = self.mesh.elements[k - 1].psi_out[g]
          t_sweepout1 += time.time() - t_sweepout1a

          t_sweepout2a = time.time()
          # ================================= Obtain inverted operator
          self.Ainv = elem_k.Agn_inv[g][n]

          # ================================= Determine inscattering source
          Q_0 = 0.0
          Q_1 = 0.0
          if lumped:
            for ell in range(0, self.L + 1):
              Q_0 += self.P_ell_mu_n[ell, n] * \
                     elem_k.scat_src_mg[ell, g]
          else:
            for ell in range(0, self.L + 1):
              Q_0 += self.P_ell_mu_n[ell, n] * \
                     elem_k.scat_src_mg_0[ell, g]
              Q_1 += self.P_ell_mu_n[ell, n] * \
                     elem_k.scat_src_mg_1[ell, g]

          # ================================= Adding sources
          elem_k.source[g] = self.source[g,k]
          qrad = self.source[g,k]/2.0

          if lumped:
            self.b[0] = (Q_0 + qrad) * elem_k.intgl_varphi[0]
            self.b[1] = (Q_0 + qrad) * elem_k.intgl_varphi[1]
          else:
            self.b[0] = Q_0 * elem_k.intgl_varphi_b[0, 0] + \
                        Q_1 * elem_k.intgl_varphi_b[0, 1]
            self.b[0] += qrad * elem_k.intgl_varphi[0]
            self.b[1] = Q_0 * elem_k.intgl_varphi_b[1, 0] + \
                        Q_1 * elem_k.intgl_varphi_b[1, 1]
            self.b[1] += qrad * elem_k.intgl_varphi[1]

          # ================================= Add upwinding sources
          if mu < 0.0:
            self.b[1] -= mu * psi_upwind
          else:
            self.b[0] -= mu * psi_upwind

          # ================================= Solve the local system
          self.psi = np.matmul(self.Ainv, self.b)

          # ================================= Store outgoing angular flux
          wn=self.w_n[n]
          if mu < 0.0:
            elem_k.psi_out[g] = self.psi[0]
            elem_k.total_psiL_outgoing[g] += wn*abs(mu)*self.psi[0]
            elem_k.total_psiR_incoming[g] += wn*abs(mu)*psi_upwind
          else:
            elem_k.psi_out[g] = self.psi[1]
            elem_k.total_psiR_outgoing[g] += wn*abs(mu)*self.psi[1]
            elem_k.total_psiL_incoming[g] += wn*abs(mu)*psi_upwind
          t_sweepout2 += time.time() - t_sweepout2a

          # ================================= Accumulate scalar flux moments
          t_sweepout3a = time.time()
          psi0 = self.psi[0]
          psi1 = self.psi[1]

          # TEMP
          # if mu>0.0:
          #   psi0 = psi_upwind
          # else:
          #   psi1 = psi_upwind
          # TEMP


          for ell in range(0, self.L + 1):
            elem_k.phi_new_mg_0[ell][g] += \
                  wn * self.P_ell_mu_n[ell, n] * psi0
            elem_k.phi_new_mg_1[ell][g] += \
                  wn * self.P_ell_mu_n[ell, n] * psi1
          t_sweepout3 += time.time() - t_sweepout3a

      t_sweepout = time.time() - t_sweepout

    # ===== Compute delta phi
    t_resid = time.time()
    delta_phi_r = self.ComputePhiRelChange(gs_i,gs_f)
    t_resid = time.time() - t_resid

    if verbose:
        print("Delta_phi_r = %g" % delta_phi_r)
        print("Set Source time = %g" % t_compb)
        print("Sweep time = %g" % t_sweepout)
        print("Sweep time/per ell grp = %g ms" % (1.0e6 * t_sweepout / (gs_f - gs_i + 1) / self.mesh.Ndiv))
        print("Upwind time = %g" % t_sweepout1)
        print("Solve time = %g" % t_sweepout2)
        print("Accumulate time = %g" % t_sweepout3)
        print("Compute Residual time = %g" % t_resid)

    return delta_phi_r / (gs_f - gs_i + 1) / self.mesh.Ndiv

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  def ComputePhiRelChange(self,gs_i,gs_f):
    delta_phi_r=0.0
    for k in range(0, self.mesh.Ndiv):
      elem_k = self.mesh.elements[k]
      for ell in range(0, self.L+1):
        for g in range(gs_i, gs_f+1):
          if (elem_k.phi_new_mg_0[ell][g]>0.0):
            delta_phi_r+=abs(elem_k.phi_new_mg_0[ell][g]- \
                             elem_k.phi_old_mg_0[ell][g])/ \
                         abs(elem_k.phi_new_mg_0[ell][g])
          if (elem_k.phi_new_mg_1[ell][g]>0.0):
            delta_phi_r+=abs(elem_k.phi_new_mg_1[ell][g]- \
                             elem_k.phi_old_mg_1[ell][g])/ \
                         abs(elem_k.phi_new_mg_1[ell][g])
          if ell==0:
            elem_k.phi_delta_m0g_0[g] = elem_k.phi_new_mg_0[ell][g]- \
                                        elem_k.phi_old_mg_0[ell][g]
            elem_k.phi_delta_m0g_1[g] = elem_k.phi_new_mg_1[ell][g]- \
                                        elem_k.phi_old_mg_1[ell][g]
            elem_k.phi_new_mg_avg[ell][g] = 0.5*(elem_k.phi_new_mg_0[ell][g] + \
                                            elem_k.phi_new_mg_1[ell][g])

    return delta_phi_r

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  def ComputeResidual(self,gs_i,gs_f):
    tot_residual = 0.0
    max_balance = 0.0
    for g in range(gs_i, gs_f+1):
      for k in range(0, self.mesh.Ndiv):
        elem_k = self.mesh.elements[k]
        mat = self.materials[elem_k.mat_id]

        # ================================== Fixed source contribution
        elem_k.residual_int_g[g] = elem_k.source[g]* \
                                   (elem_k.intgl_varphi[0]+elem_k.intgl_varphi[1])

        # ================================== Inscattering source contribution
        in_scat = 0.0
        for gprime in range(0,self.G):
          in_scat += mat.sigma_s_mom[0][gprime,g]* \
                     elem_k.phi_new_mg_0[0][gprime]*elem_k.intgl_varphi[0]
          in_scat += mat.sigma_s_mom[0][gprime, g]* \
                     elem_k.phi_new_mg_1[0][gprime]*elem_k.intgl_varphi[1]

        elem_k.residual_int_g[g] += in_scat

        # ================================== Absorption contribution
        elem_k.residual_int_g[g] -= mat.sigma_t[g]* \
                                    elem_k.phi_new_mg_0[0][g]*elem_k.intgl_varphi[0]
        elem_k.residual_int_g[g] -= mat.sigma_t[g]* \
                                    elem_k.phi_new_mg_1[0][g]*elem_k.intgl_varphi[1]

        # elem_k.residual_tot_g[g] = elem_k.residual_int_g[g]/elem_k.h

        # ================================== Net surface streaming
        net_incoming = elem_k.total_psiL_incoming[g]+elem_k.total_psiR_incoming[g]
        net_outgoing = elem_k.total_psiL_outgoing[g]+elem_k.total_psiR_outgoing[g]
        elem_k.residual_tot_g[g] += net_incoming
        elem_k.residual_tot_g[g] -= net_outgoing

        # ================================== Compute avg flux based surface residual
        # ======================== Center elements
        if (k>0) and (k<(self.mesh.Ndiv-1)):
          elem_km1 = self.mesh.elements[k-1]
          elem_kp1 = self.mesh.elements[k+1]
          elem_k.residual_s_0_g[g] = elem_k.phi_new_mg_avg[0][g] - \
                                     elem_km1.phi_new_mg_avg[0][g]
          elem_k.residual_s_1_g[g] = elem_kp1.phi_new_mg_avg[0][g]- \
                                     elem_k.phi_new_mg_avg[0][g]

        # ======================== Left boundary element
        if k==0:
          elem_kp1=self.mesh.elements[k+1]
          elem_k.residual_s_1_g[g] = elem_kp1.phi_new_mg_avg[0][g] - \
                                     elem_k.phi_new_mg_avg[0][g]
          if (self.bcs[TYPEDEF][LEFT] == ISOTROPIC):
            elem_k.residual_s_0_g[g] = elem_k.phi_new_mg_avg[0][g] - \
                                       self.bcs[GROUPVAL_LEFT][g]
          else:
            elem_k.residual_s_0_g[g] = elem_k.phi_new_mg_avg[0][g]

        # ======================== Right boundary element
        if (k==(self.mesh.Ndiv-1)):
          elem_km1=self.mesh.elements[k-1]
          elem_k.residual_s_0_g[g]=elem_k.phi_new_mg_avg[0][g]- \
                                   elem_km1.phi_new_mg_avg[0][g]
          if (self.bcs[TYPEDEF][RIGHT]==ISOTROPIC):
            elem_k.residual_s_1_g[g]=self.bcs[GROUPVAL_RIGHT][g]- \
                                     elem_k.phi_new_mg_avg[0][g]
          else:
            elem_k.residual_s_1_g[g]=-elem_k.phi_new_mg_avg[0][g]

        # ======================== Contributing to total
        net_current = abs(net_incoming - net_outgoing)

        tot_residual += (elem_k.residual_tot_g[g])*(elem_k.residual_tot_g[g])

        if ((abs(elem_k.residual_tot_g[g])/(net_current+elem_k.source[g]))>max_balance):
          max_balance = abs(elem_k.residual_tot_g[g])/(net_current+elem_k.source[g])

    return math.sqrt(tot_residual),max_balance