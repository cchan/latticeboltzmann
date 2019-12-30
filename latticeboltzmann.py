#!/usr/bin/env python3
#import os
#os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
import sys
import time
import math
import cv2
from itertools import count

dtype = np.float32

# D2Q9 https://arxiv.org/pdf/0908.4520.pdf # Normalized boltzmann distribution (thermal)
r2 = 3
w = np.array([1/36, 1/9, 1/36,
              1/9,  4/9, 1/9,
              1/36, 1/9, 1/36], dtype=dtype)
assert(np.all(w == np.flip(w, axis=0)))
assert(math.isclose(sum(w), 1, rel_tol=1e-6))
e = np.array([[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]])
assert(np.all(e == -np.flip(e, axis=0)))
assert((np.sum(e, axis=0) == [0, 0]).all())
e_f = np.asarray(e, dtype=dtype)

# N rows. M cells in each row.
N = 300 # rows
M = 100 # columns
OMEGA = 0.3 # affects viscosity (0 is completely viscous, 1 is zero viscosity)
p_ambient = 100 # density
u_ambient = [0, 0.10] # velocity
p_insides = 100
u_insides = [0, 0]
def isBlocked(y, x):
  return (10 <= x < M-10 and 10 <= y < N-10) and not \
         (13 <= x < M-13 and 10 <= y < N-13)
  #return (x - N/2) ** 2 + (y - N/2) ** 2 <= (N/25)**2
isBlocked = np.vectorize(isBlocked)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./latticeboltzmann.mp4', fourcc, 60, (M, N))

# distribution of velocities in a single cell at thermal equilibrium

def getEquilibrium(velocity, density):
  eu = velocity @ e_f.T # relative importance of each available direction, by dot product
  #print(eu.shape, np.square(eu).shape)
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(np.square(velocity),axis=-1),-1))

# Initialize to a thermally stable continuous flow field, and set the omega (viscosity) values.
# Note that isBlocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.
surroundings = getEquilibrium(np.array([u_ambient], dtype=dtype), np.array([p_ambient], dtype=dtype))[0]
assert(np.isclose(sum(surroundings), p_ambient)) # Conservation of mass
assert(np.isclose(surroundings @ e_f / p_ambient, u_ambient).all()) # Conservation of momentum
insides = getEquilibrium(np.array([u_insides], dtype=dtype), np.array([p_insides], dtype=dtype))[0]
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
  h = np.arctan2(u[...,1], u[...,0])/2/math.pi*180
  s = np.sum(u**2, axis=-1)**0.5*256*1000
  v = p
  #print(h.shape, s.shape, v.shape)
  #r = cv2.normalize(np.arctan2(u[...,1], u[...,0]), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #g = cv2.normalize(-np.arctan2(u[...,1], u[...,0]), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #b = np.zeros(p.shape, dtype=np.uint8)
  # Hue is 0-179, Saturation and Value are 0-255
  video.write(cv2.cvtColor(cv2.merge([
    np.asarray(np.clip(h, 0, 179), dtype=np.uint8),
    np.asarray(np.clip(s, 0, 255), dtype=np.uint8),
    np.asarray(np.clip(v, 0, 255), dtype=np.uint8)
    ]), cv2.COLOR_HSV2BGR))
  #video.write(cv2.merge([r,g,b]))

def stream(cells):
  for k, (dy, dx) in enumerate(e):
    # TODO: Cache locality: move k to be the first dimension. (but how is cache locality for getEquilibrium?)
    cells[max(dy,0):N+min(dy,0), max(dx,0):M+min(dx,0), k] = cells[max(-dy,0):N+min(-dy,0), max(-dx,0):M+min(-dx,0), k]
    cells[:, min(dx,0):max(dx,0),k] = surroundings[k]
    cells[min(dy,0):max(dy,0), :,k] = surroundings[k]

def reflect(cells):
  cells[blocked] = np.flip(cells[blocked], axis=-1)
  #cells[0:3,:,:] = getEquilibrium(np.array([[0,0.2]]), [p_ambient])[0]


cells = np.where(np.expand_dims(blocked,-1), np.array(0,ndmin=3), np.array(insides, ndmin=3)) # cells should have k as its first dimension for cache efficiency
stream(cells)
reflect(cells)
cells = np.where(np.expand_dims(blocked,-1), cells, np.array(insides, ndmin=3))

try:
  for iter in count():
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

except KeyboardInterrupt:
  print("Done")
  video.release()
