# Hybrid Noise Reduction

A C++ implementation of a two-stage mesh denoising method used in my bachelor’s thesis:
Diffusion — anisotropic diffusion to smooth the mesh and eliminate high-frequency noise 
Optimization — modified moving least squares (MLS) to approximate local geometry and reduce remaining low-frequency noise

The code also contains two traditional filtering methods (Laplacian filtering and bilateral filtering) for comparison and a Chamfer distance to evaluate the algorithms.

## Prerequisites

- A C++17 compiler  
- CMake ≥ 3.10  
- pmp-library 

## Build

```bash
# Download the repository
git clone --recursive https://github.com/RomanJancich1/hybrid-noise-reduction.git
cd hybrid-noise-reduction

# Create and enter build directory
mkdir build
cd build

# Configure and compile
cmake ..
cmake --build .
```
