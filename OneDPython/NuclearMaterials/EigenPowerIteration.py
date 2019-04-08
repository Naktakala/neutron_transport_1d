import numpy as np


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Eigen value power iteration
def Eigen_PI(A):
  n=np.shape(A)[0]
  y=np.ones(n)

  Ay=np.matmul(A, y)
  lamb=np.dot(y, Ay)
  y=Ay/np.linalg.norm(Ay)

  if lamb<0:
    lamb=lamb*(-1.0)

  for i in range(0, 1000):
    lamb0=lamb
    Ay=np.matmul(A, y)
    lamb=np.dot(y, Ay)
    y=Ay/np.linalg.norm(Ay)
    if lamb<0:
      lamb=lamb*(-1.0)

    if abs(lamb-lamb0)<1.0e-12:
      print("Power iteration converged after %d iterations"%i)
      break

  y=Ay/np.linalg.norm(Ay)
  y/=np.sum(y)

  return lamb, y
