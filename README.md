# Quantum Dynamics Simulation (qd)

A Python-based computational physics package for simulating quantum dynamics and computing linear and non-linear spectra using the pure state method of propagating Ehrenfest dynamics.

## Features

- Implements Ehrenfest dynamics for quantum system simulations
- Supports multiple types of harmonic oscillator baths (Debye, Ohmic, Holstein)
- Optimized for performance using Numba and NumPy
- Handles wavefunction propagation and bath interactions
- Computes both linear and non-linear spectra

## Key Components

- `frenkel.py`: Contains methods for constructing and managing harmonic oscillator baths, including spectral density discretization
- `response_hbar.py`: Implements core functionality for wavefunction propagation and Ehrenfest dynamics

## Requirements

- Python 3.x
- NumPy
- Numba

## Usage

The package is designed for simulating quantum systems coupled to their environment, particularly useful for spectroscopic calculations and quantum dynamics studies.

## License

[Add appropriate license information here]
