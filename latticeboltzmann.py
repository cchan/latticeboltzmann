#!/usr/bin/env python3
import numpy as np
import sys
import time
import math

w = np.array([1.0/36, 1.0/9, 1.0/36,
              1.0/9, 4.0/9, 1.0/9,
              1.0/36, 1.0/9, 1.0/36], dtype=np.float32) # Normalized boltzmann distribution (thermal)
e = np.array([[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]])
e_f = np.array(e, dtype=np.float32)

N = 45 # rows
M = 75 # columns
OMEGA = 0.2 # affects viscosity
u_ambient = [0, 0.4] # velocity
p_ambient = 0.3 # density


# distribution of velocities in a single cell at thermal equilibrium
def getEquilibrium(velocity, density):
  velocity = np.asarray(velocity, dtype=np.float32)
  density = np.asarray(density, dtype=np.float32)
  eu = velocity @ e_f.T # relative importance of each available direction
  return np.expand_dims(density,-1) * w * (1 + 3 * eu + 9/2*np.square(eu) - 3/2*np.expand_dims(np.sum(velocity*velocity,axis=-1),-1))

def isBlocked(y, x):
  return (x - 22.5) ** 2 + (y - 22.5) ** 2 <= 7.5**2

# Initialize to a thermally stable continuous flow field, and set the omega values.
# Note that isBlocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.

# N rows. M cells in each row.

surroundings = getEquilibrium([u_ambient], [p_ambient])[0]
blocked = np.fromfunction(isBlocked, (N, M))
cells = np.where(np.expand_dims(blocked,-1), np.array(0,ndmin=3), np.array(surroundings, ndmin=3))
omega = np.where(np.expand_dims(blocked,-1), np.array(1,ndmin=3), np.array(OMEGA, ndmin=3))

#sys.stdout.write("\033[2J")

iters = 0
while True:
  # Collisions (decay toward boltzmann distribution)
  u = cells @ e_f # net velocity in this cell
  p = np.sum(cells, axis=2) # total density in this cell
  equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution
  cells = (1-omega) * equilibrium + omega * cells # decay toward thermal equilibrium

  # Streaming (movement)
  for k, (dx, dy) in enumerate(e):
    if dx > 0 and dy > 0:
      cells[:,:,k] = np.pad(cells[:-dx,:-dy,k], ((dx, 0), (dy, 0)), mode='constant', constant_values=surroundings[k])
    elif dx <= 0 and dy > 0:
      cells[:,:,k] = np.pad(cells[-dx:,:-dy,k], ((0, -dx), (dy, 0)), mode='constant', constant_values=surroundings[k])
    elif dx > 0 and dy <= 0:
      cells[:,:,k] = np.pad(cells[:-dx,-dy:,k], ((dx, 0), (0, -dy)), mode='constant', constant_values=surroundings[k])
    else: # dx <= 0 and dy <= 0:
      cells[:,:,k] = np.pad(cells[-dx:,-dy:,k], ((0, -dx), (0, -dy)), mode='constant', constant_values=surroundings[k])

  # Set objects
  cells = np.where(np.expand_dims(blocked,-1), np.flip(cells, axis=-1), cells)

  """
  # Display density field
  s = "\033[H"
  for y in range(len(cells)):
    for x in range(len(cells[y])):
      density = int(np.sum(cells[y][x])*10)
      try:
        greyscale = " .:-=+*#%@"
        s += greyscale[min(density, len(greyscale)-1)] + greyscale[min(density, len(greyscale)-1)]
      except IndexError:
        print(density, cells[y][x])
        raise
    s += "\n"
  s += "\n"

  # Display velocity direction field
  for y in range(len(cells)):
    for x in range(len(cells[y])):
      velocity = cells[y][x] @ e_f
      tan = velocity[1]/velocity[0]
      
      s2 = math.sqrt(2)
      if -(s2-1) < tan <= s2-1:
        s += "--"
      elif s2-1 < tan <= s2+1:
        s += "\\\\"
      elif -(s2+1) < tan <= -(s2-1):
        s += "//"
      else:
        s += "||"
    s += "\n"
  s += "\n"
  """
  """
  for y in range(len(cells)):
    for x in range(len(cells[y])):
      velocity = cells[y][x] @ e_f
      angle = math.atan(velocity[1]/velocity[0])
      if math.isnan(angle):
        angle = math.pi/2
      s += ("0123456789"[int(angle*3 + 5)])*2
    s += "\n"
  s += "\n"
  """
  #print(s)
  iters += 1
  #print(iters)
  if iters == 10000:
    break
