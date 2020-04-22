import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np

N = 300
M = 800

with open("lb_cuda_kernel.cu", "r") as cu:
    mod = SourceModule(cu.read(), no_extern_c=1)
fused_collide_stream = mod.get_function("fused_collide_stream")

cells = np.random.randn(N, M, 3, 3).astype(np.float32)
newcells = np.empty_like(cells)
blocked = np.random.choice([False, True], (N, M))
surroundings = np.random.randn(3, 3).astype(np.float32)

for i in range(100):
    fused_collide_stream(
        drv.Out(newcells), drv.In(cells), drv.In(blocked), drv.In(surroundings),
        block=(2, N, 1), grid=(M//2, 1, 1))

print(newcells)
