# C++ / SYCL Parallel Geochemical Solver Backend

A high-performance, hardware-agnostic C++ / SYCL geochemical solver backend designed for massive multi-cell parallelism on heterogeneous accelerators (GPUs, multi-core CPUs) using [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp).

The solver focuses exclusively on simulating **equilibrium (aqueous speciation, mineral precipitation/dissolution)** and **kinetic** processes across spatial cells, mirroring the physical formulation of PHREEQC and the reference Python prototype.

---

## 🚀 Key Highlights

- **Zero-Allocation Device Kernels**: Registers and private stack frames (`FixedMatrix`, `FixedVector`) are used for all local linear algebra and Newton-Raphson speciation solves. No dynamic heap allocations inside SYCL kernels.
- **AdaptiveCpp SSCP Dynamic Function JIT**: PHREEQC Basic kinetic rate scripts translated to C++ device functions are JIT-specialized and inlined at runtime via `sycl::AdaptiveCpp_jit::dynamic_function_config` without kernel recompilation.
- **Decoupled Frontend / Backend Architecture**: Pure serial C++ Data Transfer Objects ([`include/dto.hpp`](include/dto.hpp)) completely isolate frontend callers and transport models from SYCL and GPU internals.
- **Full Verification**: Validated against standard PHREEQC benchmarks (`python/verify.out`), matching pH, element totals, and kinetic mineral dissolution down to 5 significant figures.
- **Throughput**: Achieves **> 31,800 cells/second** on 10,000 parallel cells.

---

## ⚡ Quick Start

### Prerequisites
- Linux (x86_64)
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (e.g. via Spack: `spack load adaptivecpp`)
- CMake >= 3.20, C++20 compliant compiler

### Build & Run
```bash
# 1. Load compiler environment
spack load adaptivecpp

# 2. Build the project
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 3. Run automated regression tests
./test_verify

# 4. Run multi-cell parallel simulation (e.g. 10,000 cells)
./geochem_sycl -db ../../database/phreeqc_kin.dat -i ../../python/verify.pqi -n 10000
```

---

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

1. [**Architecture & Data Interchange (`docs/architecture.md`)**](docs/architecture.md)  
   Frontend-backend decoupling, Data Transfer Objects (DTOs), lifecycle management, and AoS $\leftrightarrow$ SoA transformation.
2. [**SYCL Kernel & Memory Model (`docs/sycl_kernel_and_memory.md`)**](docs/sycl_kernel_and_memory.md)  
   Zero-allocation kernel design, Unified Shared Memory (`sycl::malloc_shared`), stack-allocated `FixedMatrix`/`FixedVector`, and thread-local partial-pivoting Gaussian elimination.
3. [**AdaptiveCpp SSCP JIT Kinetics (`docs/adaptivecpp_jit_rates.md`)**](docs/adaptivecpp_jit_rates.md)  
   Runtime dynamic function specialization, LLVM SSCP pass inlining, symbol reflection, translation of PHREEQC Basic rate equations, and registering custom rate laws.
4. [**Numerical Methods & Geochemical Formulation (`docs/numerical_methods.md`)**](docs/numerical_methods.md)  
   Thermodynamic equilibrium equations, extended Debye-Hückel activity coefficients, analytical Jacobian construction, two-stage initial equilibration workflow, and adaptive sub-stepping kinetics.
5. [**Parser & Database Subsystem (`docs/parser_and_database.md`)**](docs/parser_and_database.md)  
   Standalone C++ parser for thermodynamic databases (`phreeqc_kin.dat`) and input scripts (`.pqi`), stoichiometric matrix assembly, and species/phase filtering.
6. [**Verification & Benchmarks (`docs/verification_and_benchmarks.md`)**](docs/verification_and_benchmarks.md)  
   Quantitative comparison against PHREEQC reference runs (`verify.out`), multi-cell thread consistency, and scaling benchmarks up to 10,000 cells.
7. [**Build & Usage Guide (`docs/build_and_usage.md`)**](docs/build_and_usage.md)  
   Step-by-step compilation guide, CLI flags, test suite execution, and a minimal C++ code example for coupling external transport models.

---

## 📂 Project Structure

```
cpp_sycl/
├── CMakeLists.txt              # CMake configuration for AdaptiveCpp SSCP
├── README.md                   # Project overview & navigation (this file)
├── docs/                       # Detailed engineering documentation
│   ├── README.md
│   ├── architecture.md
│   ├── sycl_kernel_and_memory.md
│   ├── adaptivecpp_jit_rates.md
│   ├── numerical_methods.md
│   ├── parser_and_database.md
│   ├── verification_and_benchmarks.md
│   └── build_and_usage.md
├── include/
│   ├── dto.hpp                 # Decoupled serial Data Transfer Objects (DTOs)
│   ├── device_types.hpp        # Zero-allocation FixedVector, FixedMatrix, and Gauss solver
│   ├── parser.hpp              # Database & PQI parser, system matrix builder
│   ├── rates.hpp               # Rate laws & AdaptiveCpp dynamic rate placeholders
│   └── solver_backend.hpp      # Parallel USM backend state & SYCL kernel launcher
├── src/
│   ├── main.cpp                # Command-line driver executable
│   ├── parser.cpp              # Parser implementation & two-stage equilibration
│   ├── rates.cpp               # Rate registry & symbol reflection registration
│   └── solver_backend.cpp      # USM state management, device speciation solver & kernel
└── tests/
    └── test_verify.cpp         # Automated verification suite against PHREEQC reference
```
