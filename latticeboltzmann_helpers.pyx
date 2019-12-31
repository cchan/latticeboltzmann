cimport numpy

import numpy as np

# is float 32 bit? DTYPE = np.float32
r2 = 3
w = np.array([1/36, 1/9, 1/36,
              1/9,  4/9, 1/9,
              1/36, 1/9, 1/36], dtype=float)
assert(np.all(w == np.flip(w, axis=0)))
assert(math.isclose(sum(w), 1, rel_tol=1e-6))
e = np.array([[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]], dtype=float)
assert(np.all(e == -np.flip(e, axis=0)))
assert((np.sum(e, axis=0) == [0, 0]).all())

cdef fused_collide_stream(float[:,:,:] cells, float[:,:,:] new_cells, int y, int x):
  # Average speed:
  float uy = sum_i(cells[y][x][i] * e[i][0])
  float ux = sum_i(cells[y][x][i] * e[i][1])
  # Total density:
  float d = sum_i(cells[y][x][i])
  # Collide it and assign new locations
  for i in range(len(e)):
    new_cells[y+e[i][0]][x+e[i][1]] = d * e[i] * (1 + r2*eu + r2*r2/2*eu*eu - r2/2*(ux*ux + uy*uy))
  also copy expired new_cells to cells

  eu = velocity @ e_f.T # relative importance of each available direction, by dot product
  #print(eu.shape, np.square(eu).shape)
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(np.square(velocity),axis=-1),-1))


# distribution of velocities in a single cell at thermal equilibrium
def getEquilibrium(velocity, density):
  eu = velocity @ e_f.T # relative importance of each available direction, by dot product
  #print(eu.shape, np.square(eu).shape)
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(np.square(velocity),axis=-1),-1))


def collide(cells, u, p):
  equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution given the net density/velocity in this cell

  # decay toward thermal equilibrium
  cells -= equilibrium
  cells *= omega
  cells += equilibrium


cdef int clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)


def compute(int[:, :] array_1, int[:, :] array_2, int a, int b, int c):

    cdef Py_ssize_t x_max = array_1.shape[0]
    cdef Py_ssize_t y_max = array_1.shape[1]

    # array_1.shape is now a C array, no it's not possible
    # to compare it simply by using == without a for-loop.
    # To be able to compare it to array_2.shape easily,
    # we convert them both to Python tuples.
    assert tuple(array_1.shape) == tuple(array_2.shape)

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:, :] result_view = result

    cdef int tmp
    cdef Py_ssize_t x, y

    for x in range(x_max):
        for y in range(y_max):

            tmp = clip(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result_view[x, y] = tmp + c

    return result


def getEquilibrium(velocity, density):
  eu = velocity @ e_f.T # relative importance of each available direction, by dot product
  #print(eu.shape, np.square(eu).shape)
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(np.square(velocity),axis=-1),-1))
