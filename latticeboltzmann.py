#!/usr/bin/env python3
#import os
#os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
import sys
import time
import math
import cv2


dtype = np.float64

# D2Q21 https://arxiv.org/pdf/0908.4520.pdf # Normalized boltzmann distribution (thermal)
r2 = 3/2
w = np.array([                1/1620,
                 1/432,       7/360,        1/432,
                        2/27, 1/12,   2/27,
         1/1620, 7/360, 1/12, 91/324, 1/12, 7/360, 1/1620,
                        2/27, 1/12,   2/27,
                 1/432,       7/360,        1/432,
                              1/1620], dtype=dtype)
assert(np.all(w == np.flip(w, axis=0)))
assert(math.isclose(sum(w), 1, rel_tol=1e-6))
e = np.array([                [0,-3],
              [-2,-2],        [0,-2],       [2,-2],
                      [-1,-1],[0,-1],[1,-1],
      [-3, 0],[-2, 0],[-1, 0],[0, 0],[1, 0],[2, 0],[3, 0],
                      [-1, 1],[0, 1],[1, 1],
              [-2, 2],        [0, 2],       [2, 2],
                              [0, 3]])
assert(np.all(e == -np.flip(e, axis=0)))
assert((np.sum(e, axis=0) == [0, 0]).all())
e_f = np.asarray(e, dtype=dtype)

# N rows. M cells in each row.
N = 400 # rows
M = 800 # columns
OMEGA = 0.01 # affects viscosity (0 is completely viscous, 1 is zero viscosity)
p_ambient = 100 # density
u_ambient = [0, 0.3] # velocity BUG: why is anything moving when velocity is [0, 0]? (or is it just near 0 and numerically unstable)
def isBlocked(y, x):
  return np.logical_and(True, (x - N/2) ** 2 + (y - N/2) ** 2 <= (N/16)**2)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./latticeboltzmann.mp4', fourcc, 60, (M, N))

# distribution of velocities in a single cell at thermal equilibrium
def getEquilibrium(velocity, density):
  eu = velocity @ e_f.T # relative importance of each available direction, by dot product
  #print(eu.shape, np.square(eu).shape)
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(velocity*velocity,axis=-1),-1))

# Initialize to a thermally stable continuous flow field, and set the omega (viscosity) values.
# Note that isBlocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.
surroundings = getEquilibrium(np.array([u_ambient], dtype=dtype), np.array([p_ambient], dtype=dtype))[0]
assert(np.isclose(sum(surroundings), p_ambient)) # Conservation of mass
assert(np.isclose(surroundings @ e_f / p_ambient, u_ambient).all()) # Conservation of momentum
blocked = np.fromfunction(isBlocked, (N, M))
omega = np.where(np.expand_dims(blocked,-1), np.array(1,ndmin=3), np.array(OMEGA, ndmin=3))


def collide(cells, u, p):
  equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution given the net density/velocity in this cell

  # decay toward thermal equilibrium
  cells -= equilibrium
  cells *= omega
  cells += equilibrium

def display(u, p, video):
  #gray = np.asarray(255-p * 500, dtype=np.uint8)
  #video.write(cv2.merge([gray, gray, gray]))
  gray = cv2.normalize(p, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  h = np.asarray(np.arctan2(u[...,1], u[...,0])/2/math.pi*256, dtype=np.uint8)
  #print(u[...,0], u[...,1], h)
  s = np.full(p.shape, 255, dtype=np.uint8)
  v = cv2.normalize(p, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #print(h.shape, s.shape, v.shape)
  #r = cv2.normalize(np.arctan2(u[...,1], u[...,0]), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #g = cv2.normalize(-np.arctan2(u[...,1], u[...,0]), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #b = np.zeros(p.shape, dtype=np.uint8)
  video.write(cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR))
  #video.write(cv2.merge([r,g,b]))

def stream(cells):
  for k, (dy, dx) in enumerate(e):
    # TODO: Cache locality: move k to be the first dimension.
    cells[max(dy,0):N+min(dy,0), max(dx,0):M+min(dx,0), k] = cells[max(-dy,0):N+min(-dy,0), max(-dx,0):M+min(-dx,0), k]
    cells[:, min(dx,0):max(dx,0),k] = surroundings[k]
    cells[min(dy,0):max(dy,0), :,k] = surroundings[k]

def reflect(cells):
  cells[blocked] = np.flip(cells[blocked], axis=-1)
  #cells[40:45,10,:] = getEquilibrium(np.array([u_ambient])*5, [p_ambient])[0]


cells = np.where(np.expand_dims(blocked,-1), np.array(0,ndmin=3), np.array(surroundings, ndmin=3)) # cells should have k as its first dimension for cache efficiency
stream(cells)
reflect(cells)
cells = np.where(np.expand_dims(blocked,-1), cells, np.array(surroundings, ndmin=3))

for iter in range(500):
  sys.stdout.write(str(iter)+' ')
  sys.stdout.flush()

  # Get the total density and net velocity for each cell
  p = np.sum(cells, axis=2) # total density in this cell
  with np.errstate(divide='ignore', invalid='ignore'):
    u = cells @ e_f / np.expand_dims(p, -1) # net velocity in this cell
  np.nan_to_num(u, copy=False) # Is this a bad hack? if p == 0 (i.e. blocked) then we want u to be zero.

  # Collisions (decay toward boltzmann distribution)
  collide(cells, u, p)
  # Display density
  display(u, p, video)
  # Streaming (movement)
  stream(cells)
  # Reflect at object edges
  reflect(cells)

print()

video.release()
