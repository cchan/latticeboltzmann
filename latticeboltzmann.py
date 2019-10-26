#!/usr/bin/env python3
import numpy as np
import sys
import time
import math
import cv2

# Verify Conservation of Energy & Momentum

# D2Q21 https://arxiv.org/pdf/0908.4520.pdf # Normalized boltzmann distribution (thermal)
r2 = 3/2
w = np.array([                1/1620,
                 1/432,       7/360,        1/432,
                        2/27, 1/12,   2/27,
         1/1620, 7/360, 1/12, 91/324, 1/12, 7/360, 1/1620,
                        2/27, 1/12,   2/27,
                 1/432,       7/360,        1/432,
                              1/1620], dtype=np.float32)
assert(np.all(w == np.flip(w, axis=0)))
e = np.array([                [0,-3],
              [-2,-2],        [0,-2],       [2,-2],
                      [-1,-1],[0,-1],[1,-1],
      [-3, 0],[-2, 0],[-1, 0],[0, 0],[1, 0],[2, 0],[3, 0],
                      [-1, 1],[0, 1],[1, 1],
              [-2, 2],        [0, 2],       [2, 2],
                              [0, 3]])
assert(np.all(e == -np.flip(e, axis=0)))
e_f = np.asarray(e, dtype=np.float32)

# N rows. M cells in each row.
N = 100 # rows
M = 200 # columns
OMEGA = 0.8 # affects viscosity
p_ambient = 0.1 # density
u_ambient = [0, 0.1] # velocity
def isBlocked(y, x):
  #return x == -10000000
  #return x == y
  return (x - N/2) ** 2 + (y - N/2) ** 2 <= (N/9)**2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./latticeboltzmann.mp4', fourcc, 60, (M, N))

# distribution of velocities in a single cell at thermal equilibrium
@profile
def getEquilibrium(velocity, density):
  eu = velocity @ e_f.T # relative importance of each available direction
  #print(eu.shape, np.square(eu).shape)
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(velocity*velocity,axis=-1),-1))

# Initialize to a thermally stable continuous flow field, and set the omega values.
# Note that isBlocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.
surroundings = getEquilibrium(np.array([u_ambient], dtype=np.float32), np.array([p_ambient], dtype=np.float32))[0]
blocked = np.fromfunction(isBlocked, (N, M))
cells = np.where(np.expand_dims(blocked,-1), np.array(0,ndmin=3), np.array(surroundings, ndmin=3))
omega = np.where(np.expand_dims(blocked,-1), np.array(1,ndmin=3), np.array(OMEGA, ndmin=3))

@profile
def loop(cells):
  # Collisions (decay toward boltzmann distribution)
  p = np.sum(cells, axis=2) # total density in this cell
  u = cells @ e_f / np.expand_dims(p, -1) # net velocity in this cell
  np.nan_to_num(u, copy=False) # ??? How do we make this more numerically stable?
  #print(u_ambient, u)
  equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution
  #print(surroundings[0], cells[:,:,0], equilibrium[:,:,0], "!")
  #break
  cells = equilibrium + omega * (cells - equilibrium) # decay toward thermal equilibrium

  # Display density
  #gray = np.asarray(255-p * 500, dtype=np.uint8)
  #video.write(cv2.merge([gray, gray, gray]))
  gray = cv2.normalize(-p, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

  h = np.asarray(np.arctan2(u[...,1], u[...,0])/2/math.pi*256, dtype=np.uint8)
  s = np.full(p.shape, 255, dtype=np.uint8)
  v = cv2.normalize(p, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #print(h.shape, s.shape, v.shape)
  #input()
  #r = cv2.normalize(np.arctan2(u[...,1], u[...,0]), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #g = cv2.normalize(-np.arctan2(u[...,1], u[...,0]), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  #b = np.zeros(p.shape, dtype=np.uint8)
  video.write(cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR))

  # Streaming (movement)
  # This step is slow because np.pad copies the array. Can we do it in place?
  for k, (dy, dx) in enumerate(e):
    #print(dy, dx)
    cells[max(dy,0):N+min(dy,0), max(dx,0):M+min(dx,0), k] = cells[max(-dy,0):N+min(-dy,0), max(-dx,0):M+min(-dx,0), k]
    cells[:, min(dx,0):M+max(dx,0),k] = surroundings[k]
    cells[min(dy,0):N+max(dy,0), :,k] = surroundings[k]

  # Set objects
  cells = np.where(np.expand_dims(blocked,-1), np.flip(cells, axis=-1), cells)
  #cells[40:45,10,:] = getEquilibrium(np.array([u_ambient])*5, [p_ambient])[0]

  return cells


for iter in range(500):
  print(iter)
  cells = loop(cells)

video.release()
