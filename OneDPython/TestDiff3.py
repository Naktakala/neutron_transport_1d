import numpy as np


ra = 0.0
rb = 1.0

Nel = 20
dr = (rb-ra)/Nel
r = np.zeros(Nel)
for i in range(0,Nel):
  r[i] = 0.5*dr + i*dr

