#pragma once

#include <string>
#include <sycl/sycl.hpp>
#include "solver_backend.hpp"
#include "dto.hpp"

namespace geochem {

class JitCompiler {
public:
    // Compiles a new dynamic library containing the backend and the selected kinetics,
    // loads it, and returns the newly instantiated IBackendSolver.
    static IBackendSolver* compile_and_load(sycl::queue q, const SystemMatrices& matrices, const std::string& kinetics_dir);
};

} // namespace geochem
