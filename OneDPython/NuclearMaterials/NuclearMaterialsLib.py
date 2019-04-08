import numpy as np
import GolubFischer

import EigenPowerIteration
import math

class NuclearMaterial:
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constructor
  def __init__(self,name,L,G):
    self.name = name
    self.A = 1.0                        # Atomic mass
    self.L = L                          # Scattering order
    self.G = G                          # Num of E-grps
    self.sigma_t = np.zeros(G)          # Total xs
    self.sigma_s = np.zeros(G)          # Total scat-xs sum_gprime sigma_s0[g,gprime]
    self.sigma_s_mom = []               # Transfer matrix
    self.sigma_s_mom_limits = []        # First, last

    self.scat_e_cdf_gp_g = []           # CDF for scattering from group g to gprime
    self.scat_angles_gp_g = []          # Discrete scattering angles for g to gprime
    self.scat_weights_gp_g= []          # Discrete weights
    self.scat_mucdf_gp_g=[]             # CDF for mu scat from g to gprime

    self.xi = []                        # Maxwellian energy shape function
    self.rho = 99.0                     # Eigen-value associated with Gauss-Seidel

    self.D_g = []                       # Diffusion coefficient
    self.sigma_r = []                   # In-group removal cross-section
    self.TG_D = 1.0                     # Two-Grid acceleration diffusion coefficient
    self.TG_siga = 0.0                  # Two-Grid acceleration absorbtion

    # =========================================== Init transfer matrix
    for ell in range(0,L+1):
      mom_transfer = np.zeros((G,G))
      mom_transfer_limits = np.zeros((G,2),dtype=np.int)
      self.sigma_s_mom.append(mom_transfer)
      self.sigma_s_mom_limits.append(mom_transfer_limits)


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize material from PDT xs-file
  def InitializeFromPDTdata(self,filename,NDens=1.0,TransfetMT="2501"):
    pdtfile = open(filename, 'r')
    auxlines = pdtfile.readlines()
    pdtfile.seek(0, 0)

    outer_line_num=-1
    while (outer_line_num<(len(auxlines)-1)):
      outer_line_num+=1

      linewords = auxlines[outer_line_num].split()

      if (len(linewords)<2):
          continue
      #==================================== Total XS
      if (linewords[0] == "MT") and (linewords[1]=="1"):
          g=-1
          line_num = outer_line_num
          while (g<self.G):
              line_num+=1

              words = auxlines[line_num].split()

              for word in words:
                  g += 1
                  self.sigma_t[g] = float(word)*NDens
                  if (g == (self.G - 1)):
                      break

              if (g == (self.G - 1)):
                  break

      #==================================== Transfer matrix
      if (linewords[0] == "MT") and (linewords[1]==(TransfetMT+',')):
          ell = int(linewords[3])
          if (ell>self.L): continue
          #print("Extracting Moment %d %d" %(ell,outer_line_num))

          g=-1
          while (g<(self.G-1)):
              words = auxlines[outer_line_num+1].split()
              g = int(words[3])
              gi = int(words[4])
              gf = int(words[5])

              if ((gi<0) or (gf<0) ):
                  print("WARNING IN MATERIAL FILE")
                  break

              self.sigma_s_mom_limits[ell][g, 0] = gi
              self.sigma_s_mom_limits[ell][g, 1] = gf

              #print("Extracting Group %d %d" %(g,outer_line_num+1))

              line_num = outer_line_num+1
              gprime = gi-1
              while (gprime<=gf):
                  line_num+=1

                  words = auxlines[line_num].split()
                  for word in words:
                      gprime+=1
                      self.sigma_s_mom[ell][gprime,g] += float(word)*NDens

                      if (gprime==gf):
                          break

                  if (gprime == gf):
                      break
              outer_line_num = line_num


    pdtfile.close()

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize WG DSA parameters
  def InitializeDSAParams(self):
    self.D_g = np.zeros(self.G)
    self.sigma_r = np.zeros(self.G)

    for g in range(0, self.G):
      sigma_s_0=0.0
      sigma_s_1=0.0

      for gprime in range(0, self.G):
        sigma_s_0+=self.sigma_s_mom[0][g, gprime]
        sigma_s_1+=self.sigma_s_mom[1][g, gprime]

      D_g_inv=3.0*(self.sigma_t[g]-0.0*sigma_s_1)

      self.D_g[g]=1.0/D_g_inv
      self.sigma_r[g]=self.sigma_t[g]-self.sigma_s_mom[0][g, g]
      # self.mesh.elements[k].sigma_r[g] = mat.sigma_t[g] - sigma_s_0
      # self.mesh.elements[k].sigma_r[g] = mat.sigma_t[g]
      if (self.sigma_r[g]<0.0):
        print("ERROR: Negative cross section in group %d"%g)
      # print("Element %d, D[%d]=%g" %(k,g,1.0/D_g_inv))

    self.InitializeTGDSAParams()

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize Two Group Acc
  def InitializeTGDSAParams(self):
    S0_glob=np.transpose(self.sigma_s_mom[0])

    A=np.zeros((self.G, self.G))
    B=np.zeros((self.G, self.G))
    for g in range(0, self.G):
      A[g, g]=self.sigma_t[g]-S0_glob[g, g]
      for gprime in range(0, g):
        B[g, gprime]=S0_glob[g, gprime]
      for gprime in range(g+1, self.G):
        B[g, gprime]=S0_glob[g, gprime]

    Ainv=np.linalg.inv(A)
    AinvB=np.matmul(Ainv, B)

    eig_val, eig_vec = EigenPowerIteration.Eigen_PI(AinvB)

    self.xi  = eig_vec
    self.rho = eig_val

    self.TG_D = 0.0
    self.TG_siga=0.0
    for g in range(0,self.G):
      self.TG_D += self.D_g[g]*self.xi[g]
      sigma_gp_g = 0.0
      for gprime in range(0,self.G):
        sigma_gp_g += self.sigma_s_mom[0][gprime,g]*self.xi[gprime]
      self.TG_siga += self.sigma_t[g]*self.xi[g] - sigma_gp_g

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize MC scattering tables
  # For each gprime,g pair there is L number of moments
  # that need to have scattering angles determined
  def InitializeScatteringTables(self):
    print("Initializing scattering tables:")
    # =============================== Compute gprime to g scattering
    for gprime in range(0,self.G):
      self.scat_e_cdf_gp_g.append(np.zeros(self.G))
      sigma_s_gp_tot = np.sum(self.sigma_s_mom[0][gprime,:])
      self.sigma_s[gprime] = sigma_s_gp_tot
      if (abs(sigma_s_gp_tot)>0.0):
        for g in range(0,self.G):
          if g==0:
            self.scat_e_cdf_gp_g[gprime][g] = self.sigma_s_mom[0][gprime,0]
          else:
            self.scat_e_cdf_gp_g[gprime][g] = (self.scat_e_cdf_gp_g[gprime][g-1] + \
                                              self.sigma_s_mom[0][gprime, g])

        self.scat_e_cdf_gp_g[gprime][:] /= sigma_s_gp_tot

      # print("Escat %d, sigma_s=%g" %(gprime,sigma_s_gp_tot))
      # print(self.sigma_s_mom[0][gprime, :])
      # print( self.scat_e_cdf_gp_g[gprime][:] )
      # print(np.sum(self.scat_e_cdf_gp_g[gprime][:]))

    # =============================== Compute angular distribution
    print("Initializing discrete scattering angles:")
    for gprime in range(0,self.G):
      gplist = []
      self.scat_angles_gp_g.append(gplist)
      gplist=[]
      self.scat_weights_gp_g.append(gplist)
      gplist=[]
      self.scat_mucdf_gp_g.append(gplist)
      for g in range(0,self.G):
        Mell = np.zeros(self.L+1)
        for ell in range(0,self.L+1):
          Mell[ell] = self.sigma_s_mom[ell][gprime,g]

        mu = []
        wn = []
        wn_cumul = []
        if abs(Mell[0])>0.0:
          mu,wn = GolubFischer.GetDiscreteScatAngles(Mell)

          # ================= Compute CDF
          wn_norm=wn/(np.sum(wn))
          n=np.size(wn_norm)
          wn_cumul=np.zeros(n)
          for i in range(0, n):
            if i==0:
              wn_cumul[i]=wn_norm[0]
            else:
              wn_cumul[i]=wn_cumul[i-1]+wn_norm[i]
        else:
          mu = np.zeros(1)
          wn = np.zeros(1)
          wn_cumul = np.zeros(1)

        self.scat_angles_gp_g[gprime].append(mu)
        self.scat_weights_gp_g[gprime].append(wn)
        self.scat_mucdf_gp_g[gprime].append(wn_cumul)

        # print("Group %d->%d:" %(gprime,g))
        # print(mu)
        # print(wn)
        # print(wn_cumul)




  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Energy sampling
  # Given a random number this function samples the
  # outgoing energy group
  def SampleScatteringEnergyGrp(self,gprime,rn):
    gout = gprime
    for g in range(0,self.G):
      # if (rn<self.scat_e_cdf_gp_g[gprime][g]) and \
      #   (self.sigma_s_mom[0][gprime, g]>0.0):
      if (rn<self.scat_e_cdf_gp_g[gprime][g]):
        gout = g
        break

    return gout

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Angle sampling
  # Given a random number this function samples the
  # outgoing angle cosine
  def SampleScatteringAngle(self, gin, gout, rn):
    mu_out = 1.0
    K = np.size(self.scat_mucdf_gp_g[gin][gout])
    for k in range(0,K):
      if rn<self.scat_mucdf_gp_g[gin][gout][k]:
        mu_out = self.scat_angles_gp_g[gin][gout][k]
        break

    return mu_out




class NuclearMaterial1GIsotropic:
  def __init__(self,name,sigma_t,sigma_s):
    self.name = name
    self.A = 1.0                              # Atomic mass
    self.L = 0                                # Scattering order
    self.G = 1                                # Num of E-grps
    self.sigma_t = np.zeros(1)                # Total xs
    self.sigma_s = np.zeros(1)                # Total scat-xs sum_gprime sigma_s0[g,gprime]
    self.sigma_s_mom = []                     # Transfer matrix
    self.sigma_s_mom_limits = []              # First, last

    for ell in range(0, self.L + 1):
        mom_transfer = np.zeros((self.G, self.G)) + sigma_s
        self.sigma_s_mom.append(mom_transfer)
        limits = np.zeros((1,2),dtype=int)
        self.sigma_s_mom_limits.append(limits)

    self.sigma_t[0] = sigma_t
    self.sigma_s[0] = sigma_s

    self.avg_mu = 0.0

  def InitializeDSAParams(self):
    self.D_g=np.zeros(self.G)
    self.sigma_r=np.zeros(self.G)

    for g in range(0, self.G):
      sigma_s_0=0.0

      for gprime in range(0, self.G):
        sigma_s_0+=self.sigma_s_mom[0][g, gprime]


      D_g_inv=3.0*(self.sigma_t[g])

      self.D_g[g]=1.0/D_g_inv
      self.sigma_r[g]=self.sigma_t[g]-self.sigma_s_mom[0][g, g]
      # self.mesh.elements[k].sigma_r[g] = mat.sigma_t[g] - sigma_s_0
      # self.mesh.elements[k].sigma_r[g] = mat.sigma_t[g]
      if (self.sigma_r[g]<0.0):
        print("ERROR: Negative cross section in group %d"%g)
      # print("Element %d, D[%d]=%g" %(k,g,1.0/D_g_inv))

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Energy sampling
  # Given a random number this function samples the
  # outgoing energy group
  def SampleScatteringEnergyGrp(self, gprime, rn):
    return gprime

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Angle sampling
  # Given a random number this function samples the
  # outgoing angle cosine
  def SampleScatteringAngle(self, gin, gout, rn):
    mu_out = rn*2.0-1.0
    self.avg_mu += mu_out


    return mu_out