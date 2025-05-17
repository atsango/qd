# Quantum Dynamics Simulation (qd)

A C++ computational physics library for simulating quantum dynamics and computing linear and non-linear spectra using the pure state method of propagating Ehrenfest dynamics.

## Features

- Implements Ehrenfest dynamics for quantum system simulations
- Supports multiple types of harmonic oscillator baths (Debye, Ohmic, Holstein)
- Optimized for performance using Eigen and modern C++
- Handles wavefunction propagation and bath interactions
- Computes both linear and non-linear spectra
- Object-oriented design with proper encapsulation
- Type-safe and efficient matrix operations

## Key Components

- `quantum_dynamics.hpp`: Main header file containing all class declarations
- `quantum_dynamics.cpp`: Implementation of quantum dynamics simulation
- `examples/example.cpp`: Example usage demonstrating library functionality

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher
- Eigen3 library (version 3.3 or higher)

## Building

1. Create a build directory and navigate into it:
   ```bash
   mkdir build
   cd build
   ```

2. Generate build files:
   ```bash
   cmake ..
   ```

3. Build the library and example:
   ```bash
   make
   ```

## Usage

The library provides a modern C++ interface for simulating quantum systems coupled to their environment. Key classes include:

- `Bath`: Manages harmonic oscillator baths and their evolution
- `System`: Handles quantum system dynamics and wavefunction propagation
- `WavefunctionManager`: Manages wavefunction calculations and normalization
- `ResponseFunction`: Calculates response functions for spectroscopic studies

Example usage can be found in `examples/example.cpp`.

## License

MIT License

Copyright (c) 2025 Quantum Dynamics Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
