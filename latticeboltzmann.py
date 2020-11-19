#!/usr/bin/env python3
#import os
#os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
import sys
import time
import math
import imageio
import matplotlib
from itertools import count
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

dtype = np.float32

if dtype == np.float16:
  rtol = 1e-2
else:
  rtol = 1e-6

# Constants for D2Q9 https://arxiv.org/pdf/0908.4520.pdf # Normalized boltzmann distribution (thermal)
r2 = 3
w = np.array([1/36, 1/9, 1/36,
              1/9,  4/9, 1/9,
              1/36, 1/9, 1/36], dtype=dtype)
assert(np.all(w == np.flip(w, axis=0)))
assert(math.isclose(sum(w), 1, rel_tol=rtol))
e = np.array([[-1,-1],[0,-1],[1,-1],
              [-1, 0],[0, 0],[1, 0],
              [-1, 1],[0, 1],[1, 1]])
assert(np.all(e == -np.flip(e, axis=0)))
assert((np.sum(e, axis=0) == [0, 0]).all())
e_f = np.asarray(e, dtype=dtype)

# Configuration.
import sys
if len(sys.argv) > 1:
  from PIL import Image
  blocked = np.array(Image.open(sys.argv[1]).convert('L')) > 128
  print(blocked.shape)
  N = blocked.shape[0]
  M = blocked.shape[1]
else:
  N = 2160
  M = 3840
  def isBlocked(y, x):
    #return (10 <= x < M-10 and 10 <= y < N-10) and not \
    #       (13 <= x < M-13 and 10 <= y < N-13)
    return (x - N/4) ** 2 + (y - N/2) ** 2 <= (N/16)**2 or \
          (x - N/2) ** 2 + (y - N/3) ** 2 <= (N/9)**2 or \
          (x - N/2) ** 2 + (y - 2*N/3) ** 2 <= (N/25)**2 or \
          (x - N) ** 2 + (y - 2*N/5) ** 2 <= (N/144)**2 or \
          (x - N) ** 2 + (y - 3*N/5) ** 2 <= (N/144)**2 or \
          (x - 3*N/2) ** 2 + (y - 2*N/5) ** 2 <= (N/144)**2 or \
          (x - 3*N/2) ** 2 + (y - 3*N/5) ** 2 <= (N/144)**2 or \
          (x - 2*N) ** 2 + (y - 2*N/5) ** 2 <= (N/144)**2 or \
          (x - 2*N) ** 2 + (y - 3*N/5) ** 2 <= (N/144)**2
  isBlocked = np.vectorize(isBlocked)
  blocked = np.fromfunction(isBlocked, (N, M))

video = imageio.get_writer('./latticeboltzmann.mp4', fps=60)


OMEGA = 0.00000000000001 # affects viscosity (0 is completely viscous, 1 is zero viscosity)
p_ambient = 100 # density
u_ambient = [0, 0.1] # velocity - higher values become more unstable
p_insides = p_ambient
u_insides = u_ambient

# distribution of velocities in a single cell at thermal equilibrium
def getEquilibrium(velocity, density):
  eu = velocity @ e_f.T # relative importance of each available direction, by dot product
  return np.expand_dims(density, -1) * w * (1 + r2 * eu + r2**2/2*np.square(eu) - r2/2*np.expand_dims(np.sum(np.square(velocity),axis=-1),-1))

# Initialize to a thermally stable continuous flow field, and set the omega (viscosity) values.
surroundings = getEquilibrium(np.array([u_ambient], dtype=dtype), np.array([p_ambient], dtype=dtype))[0]
assert(np.isclose(sum(surroundings), p_ambient, rtol=rtol)) # Conservation of mass
assert(np.isclose(surroundings @ e_f / p_ambient, u_ambient, rtol=rtol).all()) # Conservation of momentum
insides = getEquilibrium(np.array([u_insides], dtype=dtype), np.array([p_insides], dtype=dtype))[0]

# Note that blocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.
omega = np.where(np.expand_dims(blocked,-1), np.array(1,ndmin=3), np.array(OMEGA, ndmin=3))


def display(u, p, video):
  h = (np.arctan2(u[...,0], u[...,1]) + math.pi)/2/math.pi
  s = np.sum(u**2, axis=-1)**0.5*100000
  v = p
  video.append_data((matplotlib.colors.hsv_to_rgb(np.clip(np.stack([h,s,v], 2), 0, 1))*255.99).astype(np.uint8))

def collide(cells, u, p):
  equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution given the net density/velocity in this cell

  # decay toward thermal equilibrium
  cells -= equilibrium
  cells *= omega
  cells += equilibrium

def stream(cells):
  for k, (dy, dx) in enumerate(e):
    # TODO: Cache locality: move k to be the first dimension. (but how is cache locality for getEquilibrium?)
    cells[max(dy,0):N+min(dy,0), max(dx,0):M+min(dx,0), k] = cells[max(-dy,0):N+min(-dy,0), max(-dx,0):M+min(-dx,0), k]
    cells[:, min(dx,0):max(dx,0),k] = surroundings[k]
    cells[min(dy,0):max(dy,0), :,k] = surroundings[k]

def reflect(cells):
  cells[blocked] = np.flip(cells[blocked], axis=-1)

with open("lb_cuda_kernel.cu", "r") as cu:
    mod = SourceModule(f"""
      #define N {N}
      #define M {M}
      #define OMEGA {OMEGA}f
    """ + ("#define half_enable\n" if dtype == np.float16 else "") + cu.read(), no_extern_c=1, options=['--use_fast_math', '-O3', '-Xptxas', '-O3,-v', '-arch', 'sm_75', '--extra-device-vectorization', '--restrict'])
fused_collide_stream = mod.get_function("fused_collide_stream")
fused_collide_stream.prepare("PPPPP")

cells = np.where(np.expand_dims(blocked,-1), np.array(insides,ndmin=3,dtype=dtype), np.array(insides, ndmin=3, dtype=dtype)) # cells should have k as its first dimension for cache efficiency
stream(cells)
reflect(cells)
cells = np.where(np.expand_dims(blocked,-1), cells, np.array(insides, ndmin=3))

cells_gpu = drv.to_device(cells)
newcells_gpu = drv.to_device(cells)
blocked_gpu = drv.to_device(blocked)
surroundings_gpu = drv.to_device(surroundings)
frame1_gpu = drv.to_device(np.empty((N, M, 3), dtype=np.uint8))
frame2_gpu = drv.to_device(np.empty((N, M, 3), dtype=np.uint8))

stream1 = drv.Stream(flags=0)
stream2 = drv.Stream(flags=0)
frame1 = drv.pagelocked_empty((N, M, 3), dtype=np.uint8)
frame2 = drv.pagelocked_empty((N, M, 3), dtype=np.uint8)

from threading import Thread
a1 = None
a2 = None
def appendData(frame, stream):
  stream.synchronize()
  video.append_data(frame)

prev_time = time.time()
try:
  for iter in range(250000):#count():
    if iter % 1000 == 999:
      # The reason it takes such a short amount of time on the first iteration is that it loads
      # a huge number of things into the stream and then hits a blocker (stream.synchronize()).
      # so the first iterations before the first render are always super fast.
      curr_time = time.time()
      print((curr_time - prev_time) * 1000, "us per iteration / ", N * M / 1000000 / (curr_time - prev_time), "GLUPS / ", N * M / 1000000 * 9 * 4 / (curr_time - prev_time), "GBps", "(iter " + str(iter+1) + ")")
      prev_time = curr_time

    # # Get the total density and net velocity for each cell
    # p = np.sum(cells, axis=2) # total density in this cell
    # with np.errstate(divide='ignore', invalid='ignore'):
    #   u = cells @ e_f / np.expand_dims(p, -1) # net velocity in this cell
    # np.nan_to_num(u, copy=False) # Is this a bad hack? if p == 0 (i.e. blocked) then we want u to be zero.

    # Fused version
    fused_collide_stream.prepared_async_call((math.ceil(M/30/4), 1, 1), (128, 1, 1), stream1,
      newcells_gpu, frame1_gpu if iter % 100 == 0 else 0, cells_gpu, blocked_gpu, surroundings_gpu)
    if iter % 200 == 0:
      if a1 is not None:
        a1.join()
      drv.memcpy_dtoh_async(frame1, frame1_gpu, stream=stream1)
      a1 = Thread(target=appendData, args=(frame1, stream1))
      a1.start()

    newcells_gpu, cells_gpu = cells_gpu, newcells_gpu
    frame1, frame2 = frame2, frame1
    frame1_gpu, frame2_gpu = frame2_gpu, frame1_gpu
    stream1, stream2 = stream2, stream1
    a1, a2 = a2, a1

    # # Display density
    # display(u, p, video)
    # # Collisions (decay toward boltzmann distribution)
    # collide(cells, u, p)
    # # Streaming (movement)
    # stream(cells)
    # # Reflect at object edges
    # reflect(cells)

except KeyboardInterrupt:
  print("Done")
  if a1 is not None: a1.join()
  if a2 is not None: a2.join()
  video.close()
