import pymp
import numpy as np

vert = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float64)
tri = np.array([[0, 1, 2]], dtype=np.int32)

print(pymp.geometry.laplacian(vert, tri).todense())
print(pymp.geometry.lumped_mass(vert, tri).todense())
print(vert)