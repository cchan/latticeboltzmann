Lattice Boltzmann Method
========================

2D lattice boltzmann fluid sim. Achieves 5.7 GLUPS on an RTX 2070, approx. 92% of maximum achievable memory bandwidth.

(Another test yields 4k630Hz)

## Nice things to look at

YouTube videos

[![yeet](https://img.youtube.com/vi/7rvzD-67sXk/0.jpg)](https://youtu.be/7rvzD-67sXk)

[![multi object stable vortex street](https://img.youtube.com/vi/H8pB7ErPXnw/0.jpg)](https://youtu.be/H8pB7ErPXnw)

[![vortex street](https://img.youtube.com/vi/Fo-gbRbTyIc/0.jpg)](https://youtu.be/Fo-gbRbTyIc)

[![lid-driven tall box vortex](https://img.youtube.com/vi/J1pS6P-js0o/0.jpg)](https://youtu.be/J1pS6P-js0o)

[![cylinder wake at high speed](https://img.youtube.com/vi/wsfL2LaHcFE/0.jpg)](https://youtu.be/wsfL2LaHcFE)

[![incorrect lattice boltzmann cfd that looks really cool](https://img.youtube.com/vi/b8ZVsETpFUE/0.jpg)](https://www.youtube.com/watch?v=b8ZVsETpFUE)

Older version screenshot (top is density, bottom is direction field):

![screenshot](screenshot.png)

## Requirements

My setup:
- AMD Ryzen 3700x
- Nvidia GeForce RTX 2070 with drivers 460.20 and CUDA 11.2
- Ubuntu 20.04 on WSL
- Python 3.7.4 using conda
I suspect but cannot test that this will work with much earlier versions / lower specs. (Was previously on 18.04 pure linux, 440.59, 10.2)

To install, just:
- `sudo apt install cuda-toolkit-11-1` (plus all the rest of https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- `pip install pycuda imageio-ffmpeg`

## Benchmarking

Achieved memory bandwidth is 406GBps, compared to 441GBps achieved in the bandwidthTest CUDA sample, and 506GBps bandwidth SOL (this was overclocked to 7899MHz; stock is 7000MHz/448GBps).

### Python profiling:
Run `python -m cProfile -s cumtime latticeboltzmann.py | less` for perhaps a minute.

Also kcachegrind with it:
  - `python -m cProfile -s tottime -o profile_data.pyprof latticeboltzmann.py`
  - `pyprof2calltree -i profile_data.pyprof -k`

### Nvidia profiling:
First allow NV usage: `echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nsight-privilege.conf` and reboot.

Then run `nv-nsight-cu-cli --target-processes all python latticeboltzmann.py`. A few seconds of samples will do.

`nvvp` is also nice, but you need to `sudo apt install openjdk-8-jdk` then `nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java`.

### Notes on overclocking:
Using MSI Afterburner, I can get a +200MHz core overclock on my 2070, yielding almost no performance boost. Overclocking the memory has much larger gains; +1100MHz overclock yields a nearly 20% performance boost.

Run the UnifiedMemoryPerf CUDA sample to get a sense of when we start encountering errors.

### Notes on nvcc:

Useful command to dump all intermediate products

```
nvcc -keep -cubin --use_fast_math -O3 -Xptxas -O3,-v -arch sm_75 --extra-device-vectorization --restrict lb_cuda_kernel.cu && cuobjdump -sass lb_cuda_kernel.cubin | grep '\/\*0' > lb_cuda_kernel.sass
```

## Future directions
- [x] Implemented in python
- [x] Very vectorized in numpy
- [ ] Javascript in-browser implementation using compute APIs
- [ ] Cython implementation
- [x] CUDA implementation using pycuda
- [ ] Julia implementation
  - With CUDANative.jl
  - With distributability
- [x] PyTorch implementation (cf https://github.com/kobejean/tf-cfd?)
  - Failed, pytorch has a 3-4x slowdown :/
- CUDA
  - [ ] Explore better simulation-time memory layouts (morton, tiling, SoA, etc. - unlikely that the display layout is the optimal computational layout)
  - [ ] D2Q21 or similar - kinda equivalent to doing two D2Q9 timesteps in one go.
  - [ ] D3Q19
  - [ ] Write newcurr directly and get rid of the double buffer... or maybe this won't help because memory again? Might get some caching benefits though.
  - [x] Try mixed-precision - implemented; 10% gain at the cost of extreme unphysical viscosity

## Resources
- http://physics.weber.edu/schroeder/javacourse/LatticeBoltzmann.pdf
- https://pdfs.semanticscholar.org/847f/819a4ea14bd789aca8bc88e85e906cfc657c.pdf
- https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

