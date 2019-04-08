import numpy as np
import matplotlib.pyplot as plt
import scipy

import PWLD_5_TGDSA

class PWLD_Methods(PWLD_5_TGDSA.PWLD_Methods):
    def WGDSA_Solve(self, gs_i, gs_f,max_iter=100,outer_iter=1):

        alpha = 0.25

        # ============================================ Initialize phi vectors
        dof_counter = -2
        for k in range(0, self.mesh.Ndiv):
            elem_k = self.mesh.elements[k]
            dof_counter += 2
            for g in range(0, self.G):
              # self.phig_dsa_new[g, dof_counter]     = elem_k.phi_delta_m0g_0[g]
              # self.phig_dsa_new[g, dof_counter + 1] = elem_k.phi_delta_m0g_1[g]
              # self.phig_dsa_new[g, dof_counter]  =elem_k.phi_new_mg_0[0][g]
              # self.phig_dsa_new[g, dof_counter+1]=elem_k.phi_new_mg_1[0][g]

              self.phig_dsa_new[g, dof_counter]  =elem_k.phi_new_mg_0[0][g] - \
                                                  elem_k.phi_old_mg_0[0][g]
              self.phig_dsa_new[g, dof_counter+1]=elem_k.phi_new_mg_1[0][g] - \
                                                  elem_k.phi_old_mg_1[0][g]

        self.phig_dsa_old = self.phig_dsa_new.copy()



        K = self.mesh.Ndiv*2
        for iter in range(0, 1):
            for g in range(gs_i, gs_f + 1):

                self.A_dsa = np.zeros((K,K))
                self.b_dsa = np.zeros(K)

                # =============================== Assemble matrix
                for k in range(0, self.mesh.Ndiv):
                    elem_k = self.mesh.elements[k]
                    mat = self.materials[elem_k.mat_id]
                    gi = mat.sigma_s_mom_limits[0][g, 0]
                    gf = mat.sigma_s_mom_limits[0][g, 1]

                    for i in range(0,2):
                        ir = 2*(k) + i
                        for j in range(0,2):
                            jr = 2*(k) + j

                            self.A_dsa[ir, jr] += mat.D_g[g] * \
                                                  elem_k.intgl_gradvarphi_gradb[i,j]
                            self.A_dsa[ir, jr] += mat.sigma_r[g] * \
                                                  elem_k.intgl_varphi_b[i, j]


                            # for gprime in range(gi, gf+1):
                            #   if gprime!=g:
                            #     self.b_dsa[ir] += mat.sigma_s_mom[0][gprime, g] * \
                            #                       self.phig_dsa_new[gprime, jr] * \
                            #                       elem_k.intgl_varphi_b[i,j]

                            gprime=g
                            self.b_dsa[ir]+=mat.sigma_s_mom[0][gprime, g]* \
                                            (self.phig_dsa_old[gprime, jr]+0.0)* \
                                            elem_k.intgl_varphi_b[i, j]

                        if (i==0):
                          self.A_dsa[ir, ir]   += alpha
                          self.A_dsa[ir, ir]   += 0.5 * mat.D_g[g] * elem_k.grad_varphi[0]
                          self.A_dsa[ir, ir+1] += 0.5 * mat.D_g[g] * elem_k.grad_varphi[1]

                          if k>0:
                            elem_km1 = self.mesh.elements[k - 1]
                            matkm1 = self.materials[elem_km1.mat_id]
                            self.A_dsa[ir, ir-1]  +=-alpha
                            self.A_dsa[ir, ir-1]  += 0.5*matkm1.D_g[g]*elem_km1.grad_varphi[1]
                            self.A_dsa[ir, ir-2]  += 0.5*matkm1.D_g[g]*elem_km1.grad_varphi[0]

                        if (i==1):
                          self.A_dsa[ir, ir]  += alpha
                          self.A_dsa[ir, ir]  +=-0.5*mat.D_g[g]*elem_k.grad_varphi[1]
                          self.A_dsa[ir, ir-1]+=-0.5*mat.D_g[g]*elem_k.grad_varphi[0]

                          if k<(self.mesh.Ndiv-1):
                            elem_kp1 = self.mesh.elements[k + 1]
                            matkp1 = self.materials[elem_kp1.mat_id]
                            self.A_dsa[ir, ir+1]+=-alpha
                            self.A_dsa[ir, ir+1]+=-0.5*matkp1.D_g[g]*elem_kp1.grad_varphi[0]
                            self.A_dsa[ir, ir+2]+=-0.5*matkp1.D_g[g]*elem_kp1.grad_varphi[1]


                phi_g,info = scipy.sparse.linalg.bicgstab(self.A_dsa, self.b_dsa,tol=1.0e-12,atol=1.0e-12)
                #phi_g = np.linalg.solve(self.A_dsa, self.b_dsa)

                for i in range(0, self.mesh.Ndiv * 2):
                    self.phig_dsa_new[g, i] = phi_g[i]

            # ============================== Compute Relative change
            res_value = 0.0
            contrib_norm = 0.0
            for g in range(gs_i, gs_f + 1 ):
                for i in range(0, self.mesh.Ndiv * 2):
                    if (abs(self.phig_dsa_new[g, i]) > 0.0):
                        res_value += abs(self.phig_dsa_new[g, i] - self.phig_dsa_old[g, i]) #/ \
                                     # abs(self.phig_dsa_new[g, i])

                    contrib_norm+=abs(self.phig_dsa_new[g, i])
                    self.phig_dsa_old[g, i] = self.phig_dsa_new[g, i]
            res_value /= (gs_f - gs_i + 1)

            print("    WGS-DSA Rel.Chg=%g L2Norm_F0=%g" % (res_value,contrib_norm))

            # ============================== Swap old phi and new phi
            for g in range(gs_i, gs_f + 1):
                for i in range(0, self.mesh.Ndiv * 2):
                    self.phig_dsa_old[g, i] = self.phig_dsa_new[g, i]
            if (res_value < 1.0e-2):
                break

        # ============================================ Push phi back
        dof_counter = -2
        for k in range(0, self.mesh.Ndiv):
            elem_k = self.mesh.elements[k]
            dof_counter += 2
            for g in range(gs_i, gs_f + 1):
              self.mesh.elements[k].phi_new_mg_0[0][g] += self.phig_dsa_new[g, dof_counter]
              self.mesh.elements[k].phi_new_mg_1[0][g] += self.phig_dsa_new[g, dof_counter + 1]


