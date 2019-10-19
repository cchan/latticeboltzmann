#!/usr/bin/env python3
import numpy as np
import sys
import time
import math

w = np.array([1.0/36, 1.0/9, 1.0/36,
              1.0/9, 4.0/9, 1.0/9,
              1.0/36, 1.0/9, 1.0/36]) # Normalized boltzmann distribution (thermal)
e = np.array([[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]])

N = 45 # rows
M = 75 # columns
OMEGA = 0.2 # affects viscosity
u_ambient = [0.4, 0] # velocity
p_ambient = 0.3 # density


# distribution of velocities in a single cell at thermal equilibrium
def getEquilibrium(velocity, density):
  velocity = np.array(velocity)
  density = np.array(density)
  eu = (e @ velocity.T).T # relative importance of each available axis
  return density[:,None] * w * (1 + 3 * eu + 9/2*np.square(eu) - 3/2*np.sum(velocity*velocity,axis=1)[:,None])

# N rows. M cells in each row.
cells = np.zeros([N*M, 9], float)
omega = np.ones(N*M)

# surrounding flow field
surroundings = getEquilibrium([u_ambient], [p_ambient])[0]

def isBlocked(i):
  x = i % M
  y = i // M
  return (x - 22.5) ** 2 + (y - 22.5) ** 2 <= 7.5**2

# Initialize to a thermally stable continuous flow field, and set the omega values.
# Note that isBlocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.
for i in range(len(cells)):
  if not isBlocked(i):
    cells[i] = surroundings
    omega[i] = OMEGA

sys.stdout.write("\033[2J")

iters = 0
while True:
  # Collisions (decay toward boltzmann distribution)
  u = cells @ e # net velocity in this cell
  p = np.sum(cells, axis=1) # total density in this cell
  equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution
  cells = (1-omega[:,None]) * equilibrium + omega[:,None] * cells # decay toward thermal equilibrium

  # Streaming (movement)
  for k, (dx, dy) in enumerate(e):
    offset = dx + M * dy
    # Cleverness in iteration order to make it doable in place in one pass
    for i in (reversed(range(len(cells))) if offset > 0 else range(len(cells)) if offset < 0 else []):
      x = i % M
      y = i // M
      if 0 <= x - dx < M and 0 <= y - dy < N:
        cells[i][k] = cells[i - offset][k]
      else:
        cells[i][k] = surroundings[k] # Outside the grid

  # Set objects
  for i in range(len(cells)):
    x = i % M
    y = i // M
    if isBlocked(i):
      cells[i] = np.flip(cells[i])

  # Display density field
  s = "\033[H"
  for i in range(len(cells)):
    density = int(np.sum(cells[i])*10)
    try:
      greyscale = " .:-=+*#%@"
      s += greyscale[min(density, len(greyscale)-1)] + greyscale[min(density, len(greyscale)-1)]
    except IndexError:
      print(density, cells[i])
      raise
    if i % M == M-1:
      s += "\n"
  s += "\n"

  # Display velocity direction field
  """
  for i in range(len(cells)):
    velocity = cells[i] @ e
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
    if i % M == M-1:
      s += "\n"
  s += "\n"
  """
  for i in range(len(cells)):
    velocity = cells[i] @ e
    angle = math.atan(velocity[1]/velocity[0])
    if math.isnan(angle):
      angle = math.pi/2
    s += ("0123456789"[int(angle*3 + 5)])*2
    if i % M == M-1:
      s += "\n"
  print(s)
  iters += 1
  print(iters)
