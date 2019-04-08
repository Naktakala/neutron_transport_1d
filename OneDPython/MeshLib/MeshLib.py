import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Element class
# This element stores information for a number
# of purposes. One of them is just simple mesh
# information and a material id.
class Element:
    def __init__(self,xi,xf):
        # ======================== Simple mesh information
        self.xi__ = xi
        self.xip1 = xf
        self.h = xf-xi
        self.mat_id = 0

        # ======================== Transport quantities
        self.source=[]
        self.phi_new_mg_0 = []
        self.phi_new_mg_1 = []
        self.phi_new_mg_avg = []
        self.phi_old_mg_0 = []
        self.phi_old_mg_1 = []
        self.phi_delta_mg_1 = []
        self.phi_delta_mg_1 = []
        self.scat_src_mg = []
        self.scat_src_mg_0 = []
        self.scat_src_mg_1 = []
        self.psi_out = []
        self.Agn_initialized=np.zeros(1)
        self.Agn_inv=[]
        self.total_psiL_incoming = []
        self.total_psiR_incoming = []
        self.total_psiL_outgoing = []
        self.total_psiR_outgoing = []

        # ======================== Residual quantities
        self.residual_int_g = [] # Interior
        self.residual_tot_g = [] # Using angular currents

        self.residual_s_0_g=[]   # Left surface using avg flux
        self.residual_s_1_g=[]   # Right surface using avg flux

        # ======================== Diffusion quantities
        self.D_g=[]
        self.sigma_r=[]

        # ======================== Finite element quantities
        self.grad_varphi = np.zeros(2)
        self.intgl_varphi = np.zeros(2)
        self.intgl_varphi_b = np.zeros((2,2))
        self.intgl_varphi_gradb = np.zeros((2,2))
        self.intgl_gradvarphi_gradb = np.zeros((2, 2))



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Mesh class
# Stores a 1D mesh for multipurpose use
# Also stores a set of PWL finite elements
class OneD_Mesh:
    def __init__(self,a,b,Ndiv):
        self.xmin = a
        self.xmax = b
        self.Ndiv = Ndiv

        self.h = (b-a)/Ndiv
        self.x = np.linspace(a,b,Ndiv+1)
        self.mat_ids = np.zeros(Ndiv)

        self.elements = []
        for k in range(0,Ndiv):
            self.elements.append(Element(self.x[k],self.x[k+1]))
