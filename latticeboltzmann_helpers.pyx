cimport numpy

import numpy as np

# is float 32 bit? DTYPE = np.float32

cdef float r2 = 3
cdef float[:] w = np.array([1/36, 1/9, 1/36,
              1/9,  4/9, 1/9,
              1/36, 1/9, 1/36])
cdef int[:,:] e = np.array([[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]])
cdef int k = 9
cdef int N = 300 # rows
cdef int M = 800 # columns
cdef float OMEGA = 0.8 # affects viscosity (0 is completely viscous, 1 is zero viscosity)

cdef fused_collide_stream(float[:,:,:] cells, float[:,:,:] equilibrium):
  cdef float uy, ux, d
  cdef int y, x, i
  for y in range(N):
    for x in range(M):
      # Average speed:
      uy = sum(cells[y][x][i] * e[i][0] for i in range(k))
      ux = sum(cells[y][x][i] * e[i][1] for i in range(k))
      # Total density:
      d = sum(cells[y][x][i] for i in range(k))
      # Collide it and assign new locations
      for i in range(k):
        eu = e[i][0] * uy + e[i][1] * ux
        equilibrium[y+e[i][0]][x+e[i][1]][i] = d * w[i] * (1 + r2*eu + r2*r2/2*eu*eu - r2/2*(ux*ux + uy*uy))

  for y in range(N):
    for x in range(M):
      for i in range(k):
        # decay toward thermal equilibrium
        cells[y][x][i] -= equilibrium[y][x][i]
        cells[y][x][i] *= OMEGA # unless blocked, in which case it should be 1?
        cells[y][x][i] += equilibrium[y][x][i]
