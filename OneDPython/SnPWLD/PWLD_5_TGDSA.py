import numpy as np
import scipy
import matplotlib.pyplot as plt

import EigenPowerIteration



class PWLD_Methods:

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize Two-Group DSA parameters
  def TGDSA_Solve(self,gs_i,gs_f):

    alpha=0.25
    K=self.mesh.Ndiv*2

    eps_tgdsa_new = np.zeros((self.G,K))
    eps_tgdsa_old = np.zeros((self.G,K))



    # =========================================== Initialize eps vectors
    dof_counter=-2
    for k in range(0, self.mesh.Ndiv):
      elem_k=self.mesh.elements[k]
      dof_counter+=2
      for g in range(0, self.G):
        eps_tgdsa_new[g, dof_counter]  =elem_k.phi_new_mg_0[0][g]- \
                                          elem_k.phi_old_mg_0[0][g]
        eps_tgdsa_new[g, dof_counter+1]=elem_k.phi_new_mg_1[0][g]- \
                                            elem_k.phi_old_mg_1[0][g]

      eps_tgdsa_old = eps_tgdsa_new.copy()

    for i in range(0,1):
      A_tgdsa=np.zeros((K, K))
      b_tgdsa=np.zeros(K)
      R_g=[]

      # =========================================== Compute R_g
      dof_counter=-2
      for k in range(0, self.mesh.Ndiv):
        elem_k=self.mesh.elements[k]
        mat = self.materials[elem_k.mat_id]
        dof_counter+=2
        for g in range(0, self.G):
          R_g.append(np.zeros(K))
          for gprime in range(g+1, self.G):
            R_g[g][dof_counter]  +=mat.sigma_s_mom[0][gprime,g]*eps_tgdsa_new[g,dof_counter]
            R_g[g][dof_counter+1]+=mat.sigma_s_mom[0][gprime,g]*eps_tgdsa_new[g,dof_counter+1]

      # =========================================== Assemble matrix
      for k in range(0, self.mesh.Ndiv):
        elem_k=self.mesh.elements[k]
        mat=self.materials[elem_k.mat_id]

        for i in range(0, 2):
          ir=2*(k)+i
          for j in range(0, 2):
            jr=2*(k)+j

            A_tgdsa[ir, jr]+=mat.TG_D* \
                             elem_k.intgl_gradvarphi_gradb[i, j]
            A_tgdsa[ir, jr]+=mat.TG_siga* \
                             elem_k.intgl_varphi_b[i, j]

            # =========================== Compute R
            for g in range(0,self.G):
              b_tgdsa[ir]+= R_g[g][jr]* \
                            elem_k.intgl_varphi_b[i, j]

          if (i==0):
            A_tgdsa[ir, ir]  += alpha
            A_tgdsa[ir, ir]  += 0.5*mat.TG_D*elem_k.grad_varphi[0]
            A_tgdsa[ir, ir+1]+= 0.5*mat.TG_D*elem_k.grad_varphi[1]

            if k>0:
              elem_km1=self.mesh.elements[k-1]
              matkm1=self.materials[elem_km1.mat_id]
              A_tgdsa[ir, ir-1] +=-alpha
              A_tgdsa[ir, ir-1] += 0.5*matkm1.TG_D*elem_km1.grad_varphi[1]
              A_tgdsa[ir, ir-2] += 0.5*matkm1.TG_D*elem_km1.grad_varphi[0]

          if (i==1):
            A_tgdsa[ir, ir]  += alpha
            A_tgdsa[ir, ir]  +=-0.5*mat.TG_D*elem_k.grad_varphi[1]
            A_tgdsa[ir, ir-1]+=-0.5*mat.TG_D*elem_k.grad_varphi[0]

            if k<(self.mesh.Ndiv-1):
              elem_kp1=self.mesh.elements[k+1]
              matkp1=self.materials[elem_kp1.mat_id]
              A_tgdsa[ir, ir+1]+=-alpha
              A_tgdsa[ir, ir+1]+=-0.5*matkp1.TG_D*elem_kp1.grad_varphi[0]
              A_tgdsa[ir, ir+2]+=-0.5*matkp1.TG_D*elem_kp1.grad_varphi[1]

      # =========================================== Solve the system
      E,info = scipy.sparse.linalg.bicgstab(A_tgdsa,b_tgdsa,tol=1.0e-12,atol=1.0e-12)
      #E=np.linalg.solve(A_tgdsa, b_tgdsa)

      # =========================================== Interpolate onto energy
      dof_counter=-2
      for k in range(0, self.mesh.Ndiv):
        elem_k=self.mesh.elements[k]
        mat = self.materials[elem_k.mat_id]
        dof_counter+=2
        for g in range(gs_i, gs_f+1):
          eps_tgdsa_new[g, dof_counter]  = E[dof_counter]*mat.xi[g]
          eps_tgdsa_new[g, dof_counter+1]= E[dof_counter+1]*mat.xi[g]

      # ============================== Compute Relative change
      res_value=0.0
      contrib_norm=0.0
      for g in range(gs_i, gs_f+1):
        for i in range(0, self.mesh.Ndiv*2):
          if (abs(eps_tgdsa_new[g, i])>0.0):
            res_value+=abs(eps_tgdsa_new[g, i]-eps_tgdsa_old[g, i])  # / \
            # abs(self.phig_dsa_new[g, i])

          contrib_norm+=abs(eps_tgdsa_new[g, i])
      res_value/=(gs_f-gs_i+1)

      print("    TG-DSA Rel.Chg=%g L2Norm_F0=%g"%(res_value, contrib_norm))

      # ============================== Swap old phi and new phi
      for g in range(gs_i, gs_f+1):
        for i in range(0, self.mesh.Ndiv*2):
          eps_tgdsa_old[g, i]=eps_tgdsa_new[g, i]
      # if (res_value<1.0e-2):
      #   break

    # ============================================ Push phi back
    dof_counter=-2
    for k in range(0, self.mesh.Ndiv):
      elem_k=self.mesh.elements[k]
      dof_counter+=2
      for g in range(gs_i, gs_f+1):
        self.mesh.elements[k].phi_new_mg_0[0][g]+=eps_tgdsa_new[g, dof_counter]
        self.mesh.elements[k].phi_new_mg_1[0][g]+=eps_tgdsa_new[g, dof_counter+1]


