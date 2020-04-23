Lattice Boltzmann Method
========================

YouTube videos

[![vortex street](https://img.youtube.com/vi/Fo-gbRbTyIc/0.jpg)](https://youtu.be/Fo-gbRbTyIc)

[![lid-driven tall box vortex](https://img.youtube.com/vi/J1pS6P-js0o/0.jpg)](https://youtu.be/J1pS6P-js0o)

[![cylinder wake at high speed](https://img.youtube.com/vi/wsfL2LaHcFE/0.jpg)](https://youtu.be/wsfL2LaHcFE)

[![incorrect lattice boltzmann cfd that looks really cool](https://img.youtube.com/vi/b8ZVsETpFUE/0.jpg)](https://www.youtube.com/watch?v=b8ZVsETpFUE)

Older version screenshot (top is density, bottom is direction field):

![screenshot](screenshot.png)

## Requirements

My setup:
- AMD Ryzen 3700x
- Nvidia GeForce RTX 2070 with drivers 440.59 and CUDA 10.2
- Ubuntu 18.04
- Python 3.7.4 using conda
I suspect but cannot test that this will work with much earlier versions / lower specs.

Install pycuda from source with `./configure.py --cuda-enable-gl`


## Benchmarking

This is really fast! I'm able to get 638fps at 3072 x 768. And yet I'm at 9% SM utilization, due to extensive memory stalls - the write pattern is pretty nonlocal so there's probably better ways to do this.

`hyperfine` is nice, from github.com/sharkdp/hyperfine.


### Python profiling:
Run `python -m cProfile -s cumtime latticeboltzmann.py | less` for perhaps a minute.

Also kcachegrind with it:
  - `python -m cProfile -s tottime -o profile_data.pyprof latticeboltzmann.py`
  - `pyprof2calltree -i profile_data.pyprof -k`

### Nvidia profiling:
First allow NV usage: `echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nsight-privilege.conf` and reboot.

Then run `nv-nsight-cu-cli --target-processes all python latticeboltzmann.py`. A few seconds of samples will do.

`nvvp` is also nice, but you need to `sudo apt install openjdk-8-jdk` then `nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java`.


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

## Resources
- http://physics.weber.edu/schroeder/javacourse/LatticeBoltzmann.pdf
- https://pdfs.semanticscholar.org/847f/819a4ea14bd789aca8bc88e85e906cfc657c.pdf
- https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

