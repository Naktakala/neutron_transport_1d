import numpy as np
import matplotlib.pyplot as plt

# This script tests 1D Diffusion using both CFEM and DFEM

LEFT = 0
RIGHT = 1
DIRICHLET = 0
VACUUM = 1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Element class
# Contains information for a 1D finite element
class Element:
  def __init__(self,xmin, xmax):
    self.h = xmax-xmin
    self.xmax = xmax
    self.xmin = xmin

    self.grad_varphi = np.zeros(2)
    self.intgl_varphi = np.zeros(2)
    self.intgl_varphi_b = np.zeros((2, 2))
    self.intgl_varphi_gradb = np.zeros((2, 2))
    self.intgl_gradvarphi_gradb = np.zeros((2, 2))

    h = self.h
    self.grad_varphi[0] = -1.0 / h
    self.grad_varphi[1] = 1.0 / h

    self.intgl_varphi[0] = h / 2.0
    self.intgl_varphi[1] = h / 2.0

    self.intgl_varphi_b[0, 0] = h / 3.0
    self.intgl_varphi_b[1, 1] = h / 3.0
    self.intgl_varphi_b[0, 1] = h / 6.0
    self.intgl_varphi_b[1, 0] = h / 6.0

    self.intgl_varphi_gradb[0, 0] = -0.5
    self.intgl_varphi_gradb[0, 1] = 0.5
    self.intgl_varphi_gradb[1, 0] = -0.5
    self.intgl_varphi_gradb[1, 1] = 0.5

    self.intgl_gradvarphi_gradb[0, 0] = 1.0 / h
    self.intgl_gradvarphi_gradb[0, 1] = -1.0 / h
    self.intgl_gradvarphi_gradb[1, 0] = -1.0 / h
    self.intgl_gradvarphi_gradb[1, 1] = 1.0 / h

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Mesh class
class Mesh:
  def __init__(self, Nel,xmin,xmax):
    self.x = np.linspace(xmin,xmax,Nel+1)
    self.Ndiv = Nel
    self.dx = (xmax - xmin)/Nel

    self.elements = []
    for k in range(0,Nel):
      self.elements.append(Element(self.x[k],self.x[k+1]))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CFEM solver
class DiffusionSolver1D_CFEM:
  def __init__(self,mesh,bcs,bcsvals):
    self.mesh=mesh
    self.bcs = bcs
    self.bcsvals = bcsvals
    self.N=self.mesh.Ndiv+1
    self.phi=np.zeros(self.N)
    self.A=np.zeros((self.N, self.N))
    self.b=np.zeros(self.N)

  # ======================================== Solve step
  def Solve(self):

    for k in range(0,self.mesh.Ndiv):
      elem_k = self.mesh.elements[k]
      alpha = 1.0/4.0
      #D=0.1
      for i in range(0,2):
        ir = k + i
        for j in range(0,2):
          jr = k + j

          #if (ir!=0) and (ir!=(self.N-1)):
          self.A[ir, jr]+=D*elem_k.intgl_gradvarphi_gradb[i, j]
          self.A[ir, jr]+=siga*elem_k.intgl_varphi_b[i, j]

        self.b[ir] += 1.0*elem_k.intgl_varphi[i]

        if (ir==0) and (self.bcs[LEFT]==VACUUM):
          #self.A[ir,ir]     +=1.0/2.0/D
          self.A[ir, ir]+=alpha
          self.A[ir, ir]+=0.5*D*elem_k.grad_varphi[0]
          self.A[ir, ir+1]+=0.5*D*elem_k.grad_varphi[1]

        if (ir==(self.N-1)) and (self.bcs[RIGHT]==VACUUM):
          #self.A[ir, ir]+=1.0/2.0/D
          self.A[ir, ir]+=alpha
          self.A[ir, ir]+=-0.5*D*elem_k.grad_varphi[1]
          self.A[ir, ir-1]+=-0.5*D*elem_k.grad_varphi[0]



    # ================================= Applying dirichlet BCS
    if self.bcs[LEFT]==DIRICHLET:
      for i in range(0,self.N):
        self.A[0,i] = 0.0
      self.A[0, 0]=1.0
      self.b[0]=self.bcsvals[LEFT]

      for j in range(1,self.N-1):
        self.b[j] -= self.A[j,0]*self.bcsvals[LEFT]
        self.A[j,0] = 0.0

    if self.bcs[RIGHT]==DIRICHLET:
      for i in range(0,self.N):
        self.A[self.N-1,i] = 0.0
      self.A[self.N-1, self.N-1]=1.0
      self.b[self.N-1]=self.bcsvals[RIGHT]

      for j in range(1,self.N-1):
        self.b[j] -= self.A[j,self.N-1]*self.bcsvals[RIGHT]
        self.A[j,self.N-1] = 0.0

    self.phi=np.linalg.solve(self.A, self.b)

    return self.mesh.x, self.phi

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DFEM solver
class DiffusionSolver1D_DFEM:
  def __init__(self,mesh,bcs,bcsvals):
    self.mesh=mesh
    self.bcs = bcs
    self.bcsvals = bcsvals
    self.N=self.mesh.Ndiv*2
    self.phi=np.zeros(self.N)
    self.x=np.zeros(self.N)
    self.A=np.zeros((self.N, self.N))
    self.b=np.zeros(self.N)

  # ======================================== Solve step
  def Solve(self):
    h=30/20
    alpha=1.0/4.0
    #D = 0.1
    for k in range(0, self.mesh.Ndiv):
      elem_k=self.mesh.elements[k]


      self.x[2*k] = elem_k.xmin
      self.x[2*k+1]=elem_k.xmax


      for i in range(0,2):
        ir = 2*k + i
        for j in range(0,2):
          jr = 2*k + j

          #if (ir!=0) and (ir!=(self.N-1)):
          self.A[ir, jr]+=D*elem_k.intgl_gradvarphi_gradb[i, j]
          self.A[ir, jr]+=siga*elem_k.intgl_varphi_b[i, j]

        self.b[ir]+=1.0*elem_k.intgl_varphi[i]

        if i==0:
          self.A[ir,ir]     +=alpha
          self.A[ir,ir]     +=0.5*D*elem_k.grad_varphi[0]
          self.A[ir,ir+1]   +=0.5*D*elem_k.grad_varphi[1]

          if (k>0):
            elem_km1 = self.mesh.elements[k-1]
            self.A[ir, ir-1]-=alpha
            self.A[ir, ir-1]+=0.5*D*elem_km1.grad_varphi[1]
            self.A[ir, ir-2]+=0.5*D*elem_km1.grad_varphi[0]

        if i==1:
          self.A[ir, ir]    +=alpha
          self.A[ir, ir]    +=-0.5*D*elem_k.grad_varphi[1]
          self.A[ir, ir-1]  +=-0.5*D*elem_k.grad_varphi[0]

          if (k<(self.mesh.Ndiv-1)):
            elem_kp1 = self.mesh.elements[k+1]
            self.A[ir, ir+1]+=-alpha
            self.A[ir, ir+1]+=-0.5*D*elem_kp1.grad_varphi[0]
            self.A[ir, ir+2]+=-0.5*D*elem_kp1.grad_varphi[1]

        # if (ir==0) and (self.bcs[LEFT]==VACUUM):
        #   self.A[ir, ir]  -=-alpha
        #   self.A[ir, ir]  -=0.5*D*elem_k.grad_varphi[0]
        #   self.A[ir, ir+1]-=0.5*D*elem_k.grad_varphi[1]
        #
        #   self.A[ir, ir] += 1.0/2.0/D
        #
        # if (ir==(self.N-1) and (self.bcs[RIGHT]==VACUUM)):
        #   self.A[ir, ir]  -=-alpha
        #   self.A[ir, ir]  -=-0.5*D*elem_k.grad_varphi[1]
        #   self.A[ir, ir-1]-=-0.5*D*elem_k.grad_varphi[0]
        #
        #   self.A[ir, ir]+=1.0/2.0/D


    # ================================= Applying dirichlet BCS
    if self.bcs[LEFT]==DIRICHLET:
      for i in range(0, self.N):
        self.A[0, i]=0.0
      self.A[0, 0]=1.0
      self.b[0]=self.bcsvals[LEFT]

      for j in range(1, self.N-1):
        self.b[j]-=self.A[j, 0]*self.bcsvals[LEFT]
        self.A[j, 0]=0.0

    if self.bcs[RIGHT]==DIRICHLET:
      for i in range(0, self.N):
        self.A[self.N-1, i]=0.0
      self.A[self.N-1, self.N-1]=1.0
      self.b[self.N-1]=self.bcsvals[RIGHT]

      for j in range(1, self.N-1):
        self.b[j]-=self.A[j, self.N-1]*self.bcsvals[RIGHT]
        self.A[j, self.N-1]=0.0


    self.phi=np.linalg.solve(self.A, self.b)
    print(self.A[0:5,0:5])

    return self.x, self.phi



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Problem
mesh = Mesh(40, 0.0, 30.0)
bcs = np.array([VACUUM,VACUUM])
bcsvals = np.array([10.0,30.0])
#FDSolver = DiffusionSolver1D_FiniteDifference(mesh,bcs,bcsvals)
CFEMSolver = DiffusionSolver1D_CFEM(mesh,bcs,bcsvals)
DFEMSolver = DiffusionSolver1D_DFEM(mesh,bcs,bcsvals)

D=0.83142106
siga=0.10846



#x1,phi1 = FDSolver.Solve()
x2,phi2 = CFEMSolver.Solve()
x3,phi3 = DFEMSolver.Solve()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plotting
plt.figure(1)
plt.clf()
#plt.plot(x1,phi1, label='Finite Difference')
plt.plot(x2,phi2, label='CFEM')
plt.plot(x3,phi3, label='DFEM')


plt.legend()
plt.show()

