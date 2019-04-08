import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

theta = math.pi/4
vaphi = math.pi/2

x_i = np.zeros(3)
omega_i = np.zeros(3)
omega_i[0] = math.sin(theta)*math.cos(vaphi)
omega_i[1] = math.sin(theta)*math.sin(vaphi)
omega_i[2] = math.cos(theta)
print(omega_i)
x_f = omega_i

theta = math.pi/2
vaphi = math.pi/2
omega_fs = np.zeros(3)
omega_fs[0] = math.sin(theta)*math.cos(vaphi)
omega_fs[1] = math.sin(theta)*math.sin(vaphi)
omega_fs[2] = math.cos(theta)

khat = np.array([0,0,1.0])
tangent = np.cross(khat,omega_i)
binorm = np.cross(tangent,omega_i)

R = np.zeros((3,3))
R[:,0] = binorm[:]
R[:,1] = tangent[:]
R[:,2] = omega_i[:]

omega_f = np.matmul(R,omega_fs) + x_f
print(R)



fig = plt.figure(0)
ax = fig.gca(projection='3d')

ax.plot([x_i[0],omega_i[0]],
        [x_i[1],omega_i[1]],
        [x_i[2],omega_i[2]])

ax.plot([x_f[0],omega_f[0]],
        [x_f[1],omega_f[1]],
        [x_f[2],omega_f[2]])
plt.xlim(-1,1)
plt.ylim(-1,1)
ax.set_zlim(0,2)
plt.show()