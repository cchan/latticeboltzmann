#!/usr/bin/env python3
import numpy as np
import sys
import time
import math
import cv2

#np.seterr(all='raise')

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
p_ambient = 10 # density
u_ambient = [0, 0.3] # velocity
def isBlocked(y, x):
  #return x == -10000000
  #return x == y
  return (x - 50) ** 2 + (y - 50) ** 2 <= 12**2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./latticeboltzmann.mp4', fourcc, 60, (M, N))

# distribution of velocities in a single cell at thermal equilibrium
def getEquilibrium(velocity, density):
  velocity = np.asarray(velocity, dtype=np.float32)
  density = np.asarray(density, dtype=np.float32)
  eu = velocity @ e_f.T # relative importance of each available direction
  #print(eu.shape, np.square(eu).shape)
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(velocity*velocity,axis=-1),-1))

# Initialize to a thermally stable continuous flow field, and set the omega values.
# Note that isBlocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.


surroundings = getEquilibrium([u_ambient], [p_ambient])[0]
print(surroundings)
print("???", u_ambient, surroundings @ e_f)
blocked = np.fromfunction(isBlocked, (N, M))
cells = np.where(np.expand_dims(blocked,-1), np.array(0,ndmin=3), np.array(surroundings, ndmin=3))
omega = np.where(np.expand_dims(blocked,-1), np.array(1,ndmin=3), np.array(OMEGA, ndmin=3))

#sys.stdout.write("\033[2J")

for iter in range(1000):
  print(iter)

  # Collisions (decay toward boltzmann distribution)
  p = np.sum(cells, axis=2) # total density in this cell
  u = cells @ e_f / np.expand_dims(p, -1) # net velocity in this cell
  np.nan_to_num(u, copy=False)
  #print(u_ambient, u)
  equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution
  #print(surroundings[0], cells[:,:,0], equilibrium[:,:,0], "!")
  #break
  cells = (1-omega) * equilibrium + omega * cells # decay toward thermal equilibrium

  # Display density
  gray = np.asarray(255-p * 500, dtype=np.uint8)
  #video.write(cv2.merge([gray, gray, gray]))
  r = gray+np.asarray(u[...,0]*500, dtype=np.uint8)
  g = gray+np.zeros(p.shape, dtype=np.uint8)
  b = gray+np.asarray(u[...,1]*500, dtype=np.uint8)
  video.write(cv2.merge([r, g, b]))
  
  # Streaming (movement)
  # This step is slow because np.pad copies the array. Can we do it in place?
  for k, (dx, dy) in enumerate(e):
    if dx > 0 and dy > 0:
      cells[...,k] = np.pad(cells[:-dx,:-dy,k], ((dx, 0), (dy, 0)), mode='constant', constant_values=(surroundings[k],))
    elif dx <= 0 and dy > 0:
      cells[...,k] = np.pad(cells[-dx:,:-dy,k], ((0, -dx), (dy, 0)), mode='constant', constant_values=(surroundings[k],))
    elif dx > 0 and dy <= 0:
      cells[...,k] = np.pad(cells[:-dx,-dy:,k], ((dx, 0), (0, -dy)), mode='constant', constant_values=(surroundings[k],))
    else: # dx <= 0 and dy <= 0:
      cells[...,k] = np.pad(cells[-dx:,-dy:,k], ((0, -dx), (0, -dy)), mode='constant', constant_values=(surroundings[k],))

  # Set objects
  cells = np.where(np.expand_dims(blocked,-1), np.flip(cells, axis=-1), cells)
  #cells[40:45:,10,:] = np.array([0,0,0,0,0,5,0,0,0]) * 10

video.release()
