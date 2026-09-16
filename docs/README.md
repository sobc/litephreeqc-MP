# C++ / SYCL Parallel Geochemical Solver Documentation

Welcome to the documentation for the C++ / SYCL parallel geochemical solver in `cpp_sycl`. This project provides a high-performance, hardware-agnostic backend for the parallel simulation of geochemical equilibrium (speciation, mineral precipitation/dissolution) and kinetic processes across spatial cells on heterogeneous architectures (multi-core CPUs, GPUs, accelerators).

---

## 📚 Documentation Index

The documentation is organized into the following topics:

1. [**Architecture & Data Interchange (`architecture.md`)**](architecture.md)  
   Frontend-backend decoupling, DTO design (`dto.hpp`), lifecycle management, dataflow, and memory serialization.
2. [**SYCL Kernel & Memory Model (`sycl_kernel_and_memory.md`)**](sycl_kernel_and_memory.md)  
   Zero-allocation device kernels, Unified Shared Memory (`sycl::malloc_shared`), register/stack allocation with `FixedMatrix`/`FixedVector`, and thread-local partial-pivoting Gaussian elimination.
3. [**AdaptiveCpp SSCP JIT Kinetics (`adaptivecpp_jit_rates.md`)**](adaptivecpp_jit_rates.md)  
   Runtime dynamic function specialization via AdaptiveCpp SSCP (`dynamic_function_config`), translating PHREEQC Basic rate scripts to C++, symbol reflection, build directory macro resolution, and registering custom rates.
4. [**Numerical Methods & Geochemical Formulation (`numerical_methods.md`)**](numerical_methods.md)  
   Newton-Raphson speciation solver, singular Jacobian avoidance via zero-mass element tracking (`has_mass[e]`), extended Debye-Hückel activity models, two-stage initial equilibration, and rate-limited adaptive sub-stepping kinetics with exact depletion handling.
5. [**Parser & Database Subsystem (`parser_and_database.md`)**](parser_and_database.md)  
   Standalone C++ parsers for PHREEQC databases (`phreeqc_kin.dat`) and multi-block PQI scripts (`.pqi`), kinetic component de-duplication, stoichiometric matrix assembly, and phase filtering.
6. [**Verification & Benchmarks (`verification_and_benchmarks.md`)**](verification_and_benchmarks.md)  
   Quantitative validation against standard PHREEQC reference runs, scaling benchmarks (> 31,800 cells/s), and large-scale 2D reactive transport verification (80,000 cells, 1,000 iterations, 3.28× chemistry speedup).
7. [**Build & Usage Guide (`build_and_usage.md`)**](build_and_usage.md)  
   Build instructions with Spack (`spack load adaptivecpp`) and CMake, CLI parameter reference, automated regression test suite (`test_verify`), and a minimal integration example for external frontends.

---

## ⚡ Quick Start

```bash
# 1. Load compiler environment (AdaptiveCpp with Clang 20)
spack load adaptivecpp

# 2. Configure build directory and compile
cd cpp_sycl
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 3. Run automated verification tests
./test_verify

# 4. Run parallel simulation with 10,000 cells
./geochem_sycl -db ../database/phreeqc_kin.dat -i ../examples/verify.pqi -n 10000
# (or with automatic default paths):
./geochem_sycl -n 10000
```
