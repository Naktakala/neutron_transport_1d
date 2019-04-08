import numpy as np
import Legendre as lg
import time

import PWLD_2_Utilities


class PWLD_Methods(PWLD_2_Utilities.PWLD_Methods):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def Initialize(self):
        print("Quadrature angles and weights:")
        self.mu_n, self.w_n = lg.LegendreRoots(self.Na)
        print(self.mu_n)
        print(self.w_n)
        print("Sum of weights: %g" % (np.sum(self.w_n)))

        print("Mesh:")
        print(self.mesh.x)

        print("Precomputing Legendre coefficients:")
        for ell in range(0, self.L + 1):
            for n in range(0, self.Na):
                self.P_ell_mu_n[ell, n] = lg.Legendre(ell, self.mu_n[n])

        print("Initializing dof array (phi)")
        for k in range(0, self.mesh.Ndiv):
            for ell in range(0, self.L + 1):
                new_phi = np.zeros(self.G)
                self.mesh.elements[k].phi_new_mg_0.append(new_phi)
                new_phi = np.zeros(self.G)
                self.mesh.elements[k].phi_new_mg_1.append(new_phi)
                new_phi = np.zeros(self.G)
                self.mesh.elements[k].phi_old_mg_0.append(new_phi)
                new_phi = np.zeros(self.G)
                self.mesh.elements[k].phi_old_mg_1.append(new_phi)
                new_phi=np.zeros(self.G)
                self.mesh.elements[k].phi_new_mg_avg.append(new_phi)

            self.mesh.elements[k].total_psiL_incoming = np.zeros(self.G)
            self.mesh.elements[k].total_psiL_outgoing = np.zeros(self.G)
            self.mesh.elements[k].total_psiR_incoming = np.zeros(self.G)
            self.mesh.elements[k].total_psiR_outgoing = np.zeros(self.G)
            self.mesh.elements[k].residual_s_0_g = np.zeros(self.G)
            self.mesh.elements[k].residual_s_1_g = np.zeros(self.G)
            self.mesh.elements[k].residual_int_g = np.zeros(self.G)
            self.mesh.elements[k].residual_tot_g = np.zeros(self.G)

            new_phi = np.zeros(self.G)
            self.mesh.elements[k].phi_delta_m0g_0=new_phi
            new_phi = np.zeros(self.G)
            self.mesh.elements[k].phi_delta_m0g_1=new_phi
            self.mesh.elements[k].source = np.zeros(self.G)

            new_scat_src = np.zeros((self.L + 1, self.G))
            new_scat_src_0 = np.zeros((self.L + 1, self.G))
            new_scat_src_1 = np.zeros((self.L + 1, self.G))
            self.mesh.elements[k].scat_src_mg = new_scat_src
            self.mesh.elements[k].scat_src_mg_0 = new_scat_src_0
            self.mesh.elements[k].scat_src_mg_1 = new_scat_src_1

            for g in range(0, self.G):
                value = 0.0
                self.mesh.elements[k].psi_out.append(value)

        print("Compute cell matrices")
        for k in range(0, self.mesh.Ndiv):
            h = self.mesh.elements[k].h

            self.mesh.elements[k].grad_varphi[0] = -1.0 / h
            self.mesh.elements[k].grad_varphi[1] = 1.0 / h

            self.mesh.elements[k].intgl_varphi[0] = h / 2.0
            self.mesh.elements[k].intgl_varphi[1] = h / 2.0

            self.mesh.elements[k].intgl_varphi_b[0, 0] = h / 3.0
            self.mesh.elements[k].intgl_varphi_b[1, 1] = h / 3.0
            self.mesh.elements[k].intgl_varphi_b[0, 1] = h / 6.0
            self.mesh.elements[k].intgl_varphi_b[1, 0] = h / 6.0

            self.mesh.elements[k].intgl_varphi_gradb[0, 0] = -0.5
            self.mesh.elements[k].intgl_varphi_gradb[0, 1] = 0.5
            self.mesh.elements[k].intgl_varphi_gradb[1, 0] = -0.5
            self.mesh.elements[k].intgl_varphi_gradb[1, 1] = 0.5

            self.mesh.elements[k].intgl_gradvarphi_gradb[0, 0] = 1.0 / h
            self.mesh.elements[k].intgl_gradvarphi_gradb[0, 1] = -1.0 / h
            self.mesh.elements[k].intgl_gradvarphi_gradb[1, 0] = -1.0 / h
            self.mesh.elements[k].intgl_gradvarphi_gradb[1, 1] = 1.0 / h



            cell_init_flags = np.zeros((self.G, self.Na), dtype=bool)
            self.mesh.elements[k].Agn_initialized = cell_init_flags.copy()
            for g in range(0, self.G):
                per_grp_angles = []
                self.mesh.elements[k].Agn_inv.append(per_grp_angles.copy())
                for n in range(0, self.Na):
                    cell_matrix = np.zeros((2, 2))
                    self.mesh.elements[k].Agn_inv[g].append(cell_matrix.copy())

            elem_k = self.mesh.elements[k]
            mat = self.materials[elem_k.mat_id]
            for n in range(0, self.Na):
                mu = self.mu_n[n]
                for g in range(0, self.G):
                    # ================================= Computing Stiffness and Mass matrix
                    self.A[0, 0] = mu * elem_k.intgl_varphi_gradb[0, 0] + \
                                   mat.sigma_t[g] * elem_k.intgl_varphi_b[0, 0]
                    self.A[0, 1] = mu * elem_k.intgl_varphi_gradb[0, 1] + \
                                   mat.sigma_t[g] * elem_k.intgl_varphi_b[0, 1]
                    self.A[1, 0] = mu * elem_k.intgl_varphi_gradb[1, 0] + \
                                   mat.sigma_t[g] * elem_k.intgl_varphi_b[1, 0]
                    self.A[1, 1] = mu * elem_k.intgl_varphi_gradb[1, 1] + \
                                   mat.sigma_t[g] * elem_k.intgl_varphi_b[1, 1]

                    if mu < 0.0:
                        self.A[1, 1] -= mu
                    else:
                        self.A[0, 0] -= mu

                    # ================================= Compute 2x2 matrix inverse
                    det = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
                    one_over_det = 1.0 / det
                    self.Ainv[0, 0] = self.A[1, 1] * one_over_det
                    self.Ainv[0, 1] = -self.A[0, 1] * one_over_det
                    self.Ainv[1, 0] = -self.A[1, 0] * one_over_det
                    self.Ainv[1, 1] = self.A[0, 0] * one_over_det

                    self.mesh.elements[k].Agn_inv[g][n] = self.Ainv.copy()

        print("Compute DSA parameters")
        # ================================== Map dsa matrix indices
        dof_counter=-2
        for k in range(0, self.mesh.Ndiv):
          elem_k=self.mesh.elements[k]
          mat=self.materials[elem_k.mat_id]
          dof_counter+=2
          self.dsa_map[dof_counter]=k
          self.dsa_map[dof_counter+1]=k

        # ================================== Initialize material DSA parameters
        for mat in self.materials:
          mat.InitializeDSAParams()

