import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

p = pcl.load_XYZRGB("PointCloud_namuga/00.ply")
p_np = np.asarray(p)
x, y, z = p_np[:,0], p_np[:, 1], p_np[:, 2]
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c = z, s= 20, alpha=0.5, cmap=plt.cm.Greens)
plt.show()
