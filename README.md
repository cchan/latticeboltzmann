Lattice Boltzmann Method
========================

I promise I'll write more stuff here about what I learned about fluid dynamics simulation when it's not 3AM.

YouTube video:

[![incorrect lattice boltzmann cfd that looks really cool](https://img.youtube.com/vi/b8ZVsETpFUE/0.jpg)](https://www.youtube.com/watch?v=b8ZVsETpFUE)

Here's a nice screenshot of an older version (top is density, bottom is direction field)

![screenshot](screenshot.png)

References:
- http://physics.weber.edu/schroeder/javacourse/LatticeBoltzmann.pdf (main one)
- https://pdfs.semanticscholar.org/847f/819a4ea14bd789aca8bc88e85e906cfc657c.pdf
- https://mikeash.com/pyblog/fluid-simulation-for-dummies.html (this is a great intro)

Stuff I've done beyond these:
- Implemented in python
- Very well vectorized though tbh could be better
- Thermalized initialization

Issues:
- Still haven't been able to replicate vortex shedding; I don't know whether it's actually a code bug or I just am not hitting the right Reynolds numbers.
- Direction field also looks wrong at first glance. I probably need better visualization as a first step.

Perftesting
- Use cProfile directly
  - `python -m cProfile -s tottime latticeboltzmann.py`
- or use kcachegrind with it
  - `python -m cProfile -s tottime -o profile_data.pyprof latticeboltzmann.py`
  - `pyprof2calltree -i profile_data.pyprof -k`

Future directions:
- Javascript in-browser implementation using compute APIs
- Cython implementation
- CUDA implementation using pycuda
- Julia implementation
  - With CUDANative.jl
  - With distributability
- PyTorch implementation (cf https://github.com/kobejean/tf-cfd?)
  - Failed, pytorch has a 3-4x slowdown :/

