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
import cv2

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
  video = imageio.get_writer(sys.argv[1] + '.mp4', fps=60)
else:
  raise ValueError("No base image provided")


if len(sys.argv) > 2:
  tune = sys.argv[2]
else:
  tune = 'RTX_3090'

if tune == 'RTX_2070':
  INNER_TIMESTEPS = 1 # Number of times the inner loop repeats.
  INNER_BLOCK = N+1 # Number of rows of a 32-wide column to be processed in an inner loop chunk.
  BLOCKS_THREADS_TUNE_CONSTANT = 8 # Adjust the blocks/threads tradeoff. Higher means more threads per block, but fewer blocks.
  NVCC_ARCH = 'sm_75'
elif tune == 'RTX_3090':
  INNER_TIMESTEPS = 1
  INNER_BLOCK = N+1
  BLOCKS_THREADS_TUNE_CONSTANT = 10
  NVCC_ARCH = 'sm_86'
elif tune == 'A100_80GB':
  INNER_TIMESTEPS = 6
  INNER_BLOCK = N+1
  BLOCKS_THREADS_TUNE_CONSTANT = 12
  NVCC_ARCH = 'sm_80'
else:
  raise NotImplementedError(f"Unknown tuning `{tune}`")
OUTPUT_INTERVAL = 201
# I think the conclusion from this tuning is:
#   1) the cache is far too small for this to work in this way (we need RDNA2 Infinity Cache for this)
#   2) there might actually be a compute bottleneck as well. which means we're at a decently optimal point.
#   3) Remember that SMs can have *thousands* of threads. On Turing the limit is 32 warps per SM * 32 threads per warp = 1024 threads per SM. Times 40 SMs in an RTX 2070. Again, the cache is far too small.
#       More detail: L1, 64 KB per SM, among 1024 threads is 64 bytes per thread. L2, 4 MB among 40 SMs, is 97.7 bytes per thread. And with fp32 we have 36 bytes per cell.
#       (This assumes max occupancy, not sure if this is true.)

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

# # Note that blocked = True means that omega is 1, so no thermal perturbation occurs before reflection back into the rest of the system.
# omega = np.where(np.expand_dims(blocked,-1), np.array(1,ndmin=3), np.array(OMEGA, ndmin=3))


# def display(u, p, video):
#   h = (np.arctan2(u[...,0], u[...,1]) + math.pi)/2/math.pi
#   s = np.sum(u**2, axis=-1)**0.5*100000
#   v = p
#   video.append_data((matplotlib.colors.hsv_to_rgb(np.clip(np.stack([h,s,v], 2), 0, 1))*255.99).astype(np.uint8))

# def collide(cells, u, p):
#   equilibrium = getEquilibrium(u, p) # get thermal equilibrium distribution given the net density/velocity in this cell

#   # decay toward thermal equilibrium
#   cells -= equilibrium
#   cells *= omega
#   cells += equilibrium

# def stream(cells):
#   for k, (dy, dx) in enumerate(e):
#     # TODO: Cache locality: move k to be the first dimension. (but how is cache locality for getEquilibrium?)
#     cells[max(dy,0):N+min(dy,0), max(dx,0):M+min(dx,0), k] = cells[max(-dy,0):N+min(-dy,0), max(-dx,0):M+min(-dx,0), k]
#     cells[:, min(dx,0):max(dx,0),k] = surroundings[k]
#     cells[min(dy,0):max(dy,0), :,k] = surroundings[k]

# def reflect(cells):
#   cells[blocked] = np.flip(cells[blocked], axis=-1)

with open("lb_cuda_kernel.cu", "r") as cu:
    prepend = f"""
      #define N {N}
      #define M {M}
      #define OMEGA {OMEGA}f
      #define INNER_TIMESTEPS {INNER_TIMESTEPS}
      #define INNER_BLOCK {INNER_BLOCK}
    """ + ("#define half_enable\n" if dtype == np.float16 else "")
    print(prepend)
    mod = SourceModule(prepend + cu.read(), no_extern_c=1, options=['-std=c++17', '--use_fast_math', '-O3', '-Xptxas', '-O3,-v,-dlcm=ca,-dscm=wt,-warn-spills,-warn-double-usage,-warn-lmem-usage', '-arch', NVCC_ARCH, '--extra-device-vectorization', '--restrict', '--resource-usage'])
fused_collide_stream = mod.get_function("fused_collide_stream")
fused_collide_stream_display = mod.get_function("fused_collide_stream_display")
fused_collide_stream.prepare("PPPP")
fused_collide_stream_display.prepare("PPPPP")
fused_collide_stream.set_cache_config(pycuda.driver.func_cache.PREFER_L1)
fused_collide_stream_display.set_cache_config(pycuda.driver.func_cache.PREFER_L1)

cells = np.where(np.expand_dims(blocked,-1), np.array(insides,ndmin=3,dtype=dtype), np.array(insides, ndmin=3, dtype=dtype)) # cells should have k as its first dimension for cache efficiency
# stream(cells)
# reflect(cells)
# cells = np.where(np.expand_dims(blocked,-1), cells, np.array(insides, ndmin=3))

cells_gpu = drv.to_device(cells)
newcells_gpu = drv.to_device(cells)
blocked_gpu = drv.to_device(blocked)
surroundings_gpu = drv.to_device(surroundings)
frame1_gpu = drv.to_device(np.empty((N, M, 3), dtype=np.uint8))
frame2_gpu = drv.to_device(np.empty((N, M, 3), dtype=np.uint8))

drv.Context.synchronize()

stream1 = drv.Stream(flags=0)
stream2 = drv.Stream(flags=0)
frame1 = drv.pagelocked_empty((N, M, 3), dtype=np.uint8)
frame2 = drv.pagelocked_empty((N, M, 3), dtype=np.uint8)

from threading import Thread
a1 = None
a2 = None
def appendData(frame, stream):
  stream.synchronize()
  video.append_data(cv2.resize(frame, dsize=(M//8, N//8), interpolation=cv2.INTER_CUBIC))

prev_time = time.time()
try:
  for curr_iter in count():
    if curr_iter % 1000 == 999:
      # The reason it takes such a short amount of time on the first iteration is that it loads
      # a huge number of things into the stream and then hits a blocker (stream.synchronize()).
      # so the first iterations before the first render are always super fast.
      curr_time = time.time()
      print((curr_time - prev_time) * 1000 / INNER_TIMESTEPS, "us per iteration / ", N * M / 1000000 / (curr_time - prev_time) * INNER_TIMESTEPS, "GLUPS / ", 2 * N * M / 1000000 * 9 * 4 / (curr_time - prev_time) * INNER_TIMESTEPS, "GBps", "(iter " + str((curr_iter+1) * INNER_TIMESTEPS) + ")")
      prev_time = curr_time

    # # Get the total density and net velocity for each cell
    # p = np.sum(cells, axis=2) # total density in this cell
    # with np.errstate(divide='ignore', invalid='ignore'):
    #   u = cells @ e_f / np.expand_dims(p, -1) # net velocity in this cell
    # np.nan_to_num(u, copy=False) # Is this a bad hack? if p == 0 (i.e. blocked) then we want u to be zero.

    # Fused version
    if curr_iter % OUTPUT_INTERVAL == 0:
      fused_collide_stream_display.prepared_async_call((math.ceil(M/(32 - 2*INNER_TIMESTEPS)/BLOCKS_THREADS_TUNE_CONSTANT), 1, 1), (32 * BLOCKS_THREADS_TUNE_CONSTANT, 1, 1), stream1,
        newcells_gpu, frame1_gpu, cells_gpu, blocked_gpu, surroundings_gpu)
      if a1 is not None:
        a1.join()
      drv.memcpy_dtoh_async(frame1, frame1_gpu, stream=stream1)
      a1 = Thread(target=appendData, args=(frame1, stream1))
      a1.start()
    else:
      fused_collide_stream_display.prepared_async_call((math.ceil(M/(32 - 2*INNER_TIMESTEPS)/BLOCKS_THREADS_TUNE_CONSTANT), 1, 1), (32 * BLOCKS_THREADS_TUNE_CONSTANT, 1, 1), stream1,
        newcells_gpu, 0, cells_gpu, blocked_gpu, surroundings_gpu)
      # For reasons I do not understand, the below non-rendering-only kernel running by itself has a substantial (>10%) performance hit
      #   compared to the above version that includes both rendering code (both cpu & gpu side) and non-rendering code.
      # fused_collide_stream.prepared_async_call((math.ceil(M/(32 - 2*INNER_TIMESTEPS)/BLOCKS_THREADS_TUNE_CONSTANT), 1, 1), (32 * BLOCKS_THREADS_TUNE_CONSTANT, 1, 1), stream1,
      #   newcells_gpu, cells_gpu, blocked_gpu, surroundings_gpu)

    newcells_gpu, cells_gpu = cells_gpu, newcells_gpu
    frame1, frame2 = frame2, frame1
    frame1_gpu, frame2_gpu = frame2_gpu, frame1_gpu
    stream1, stream2 = stream2, stream1
    a1, a2 = a2, a1

except KeyboardInterrupt:
  print("KeyboardInterrupt")
  print("current iteration: ", (curr_iter+1) * INNER_TIMESTEPS)
  pass

print("Waiting for streams and threads to finish...")
stream1.synchronize()
stream2.synchronize()
if a1 is not None: a1.join()
if a2 is not None: a2.join()
video.close()
print("Done")
