# Build & Usage Guide

## 1. Prerequisites & Environment

Building and executing this project requires a Linux environment equipped with the **AdaptiveCpp** toolchain.

### 1.1 Loading the Compiler via Spack
AdaptiveCpp is pre-installed via Spack. Before configuring or compiling, load the environment module:

```bash
spack load adaptivecpp
```

To verify the installation:
```bash
acpp --version
```
*(The output should indicate AdaptiveCpp 25.10.0 backed by Clang 20.x).*

---

## 2. Project Compilation with CMake

The project features a standalone CMake configuration in [`cpp_sycl/CMakeLists.txt`](file:///mnt/nfsshare/home/mluebke/phreeqc3/cpp_sycl/CMakeLists.txt).

### 2.1 Essential CMake Flags
To enable the AdaptiveCpp SSCP LLVM JIT pass to export and optimize device symbols properly, `CMakeLists.txt` sets the following variables **before** `find_package(AdaptiveCpp)`:
```cmake
set(ACPP_TARGETS "generic" CACHE STRING "" FORCE)
set(ACPP_EXTRA_ARGS "-O3 --acpp-targets=generic --acpp-export-all" CACHE STRING "" FORCE)
```
- `--acpp-targets=generic`: Emits portable LLVM bitcode for the generic SSCP runtime.
- `--acpp-export-all`: Retains and exports all device symbols for runtime dynamic function specialization.
- `-O3`: Activates full LLVM optimization passes.

### 2.2 Compilation Steps

```bash
cd cpp_sycl
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

The build produces three primary artifacts inside `build/`:
- `libsycl_solver_backend.so`: The shared library containing the parallel solver backend.
- `geochem_sycl`: The standalone command-line driver executable.
- `test_verify`: The automated verification test suite.

---

## 3. Running the CLI Driver (`geochem_sycl`)

`geochem_sycl` provides a flexible command-line interface for running multi-cell simulations:

```bash
./geochem_sycl -db <path-to-database> -i <path-to-pqi-script> -n <number-of-cells>
```

### Options:
- `-db <path>`: Path to thermodynamic database file (default: `../database/phreeqc_kin.dat`).
- `-i <path>`: Path to simulation input script (default: `../examples/verify.pqi`).
- `-n <int>`: Number of cells to simulate in parallel (default: `1`).

### Examples:

#### Single-Cell Run with Detailed State Output:
```bash
./geochem_sycl -db ../database/phreeqc_kin.dat -i ../examples/verify.pqi -n 1
# or simply using defaults:
./geochem_sycl
```
*Output:*
```
=========================================================
  C++ SYCL Parallel Geochemical Solver (AdaptiveCpp)
=========================================================
Target Device: AdaptiveCpp OpenMP host device
Database:      ../database/phreeqc_kin.dat
Input Script:  ../examples/verify.pqi
Cells Count:   1

Chemical System:
  Elements: Ca+2 CO3-2 SO4-2 H+ 
  Equilibrium Minerals: Gypsum 
  Kinetics: Calcite 

Performing two-stage initial equilibration...
Equilibrated Initial pH: 7.00000

Launching SYCL parallel kernel (dt = 100.00000 s across 1 cells)...
Simulation completed in 0.0226 seconds.
Throughput: 44.2 cells/sec

--- Final Simulation State (Cell 0) ---
Status: CONVERGED
Final pH: 7.00502

Elements (mol/kgw):
  Ca+2     : 1.002811e-03
  CO3-2    : 2.002811e-03
  SO4-2    : 3.000000e-03

Equilibrium Minerals (mol):
  Gypsum   : 0.000000e+00

Kinetics (mol):
  Calcite  : 9.997189e-03
=========================================================
```

#### Massive Multi-Cell Simulation (10,000 Cells):
```bash
./geochem_sycl -n 10000
```
*Output:*
```
Simulation completed in 0.3144 seconds.
Throughput: 31803.0 cells/sec
```

---

## 4. Running the Test Suite (`test_verify`)

The regression test can be invoked from either the `build/` directory or the repository root:

```bash
./test_verify
```

The test validates:
1. Parsing of the thermodynamic database and PQI script.
2. Construction of stoichiometric system matrices.
3. Two-stage initial equilibration.
4. Parallel SYCL kernel execution over 10 concurrent cells.
5. Numerical accuracy against `python/verify.out`.
6. Multi-cell thread consistency (identical results across all cells).

---

## 5. Integrating with an External Frontend

To link the SYCL backend into an external transport model or custom application:

### 5.1 Minimal C++ Example:
```cpp
#include "dto.hpp"
#include "parser.hpp"
#include "solver_backend.hpp"

int main() {
    // 1. Parse database and scenario input
    geochem::PhreeqcDatabase db;
    db.parse("database/phreeqc_kin.dat");
    geochem::PqiParser pqi;
    pqi.parse_pqi("python/verify.pqi", db);

    // 2. Build system matrices and input DTO (e.g. for 500 spatial cells)
    auto matrices = geochem::SystemBuilder::build_system_matrices(db, pqi);
    auto input = geochem::SystemBuilder::create_system_input(db, pqi, matrices, 500);

    // 3. Initialize SYCL queue & solver backend
    sycl::queue q{sycl::default_selector_v};
    geochem::BackendSolver solver(q, matrices);

    // 4. Perform initial equilibration & warm-starting
    geochem::SystemBuilder::equilibrate_initial_state(solver, input);

    // 5. Simulation time loop (e.g., coupled with physical transport)
    for (int step = 0; step < 10; ++step) {
        // Execute 100-second geochemical reaction step across all cells in parallel
        geochem::SystemOutput output = solver.run_simulation(100.0);

        // Process results
        for (const auto& cell_res : output.cells) {
            double final_ca = cell_res.element_totals.at("Ca+2");
            double final_ph = cell_res.final_ph;
            // ... perform spatial advection/dispersion step ...
        }
    }
    return 0;
}
```

### 5.2 CMake Integration:
```cmake
add_executable(my_transport_app main.cpp)
target_link_libraries(my_transport_app PRIVATE sycl_solver_backend)
target_include_directories(my_transport_app PRIVATE cpp_sycl/include)
add_sycl_to_target(TARGET my_transport_app SOURCES main.cpp)
```
