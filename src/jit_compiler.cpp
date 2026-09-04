#include "jit_compiler.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <dlfcn.h>
#include <cstdlib>
#include "basic_transpiler.hpp"

namespace geochem {

IBackendSolver* JitCompiler::compile_and_load(sycl::queue q, const SystemMatrices& matrices, const std::string& kinetics_dir) {
    std::string cache_dir = ".cache";
    if (!std::filesystem::exists(cache_dir)) {
        std::filesystem::create_directory(cache_dir);
    }

    // Create a hash/name based on the requested kinetics
    std::string hash_name = "solver";
    for (const auto& k : matrices.kinetics_names) {
        hash_name += "_" + k;
    }

    std::string so_path = cache_dir + "/" + hash_name + ".so";
    std::string cpp_path = cache_dir + "/" + hash_name + ".cpp";

    std::string include_dir = std::filesystem::absolute("include").string();
    if (!std::filesystem::exists(include_dir)) include_dir = std::filesystem::absolute("../include").string();
    std::string src_dir = std::filesystem::absolute("src").string();
    if (!std::filesystem::exists(src_dir)) src_dir = std::filesystem::absolute("../src").string();

    // If the library doesn't exist, generate and compile it
    if (!std::filesystem::exists(so_path)) {
        std::cout << "[JIT] Generating and compiling kernel: " << hash_name << " ...\n";
        
        std::ofstream out(cpp_path);
        
        // Write transpiled rate functions
        for (const auto& k : matrices.kinetics_names) {
            auto it = matrices.rates_scripts.find(k);
            if (it != matrices.rates_scripts.end()) {
                std::cout << "[JIT] Transpiling BASIC rate for " << k << "...\n";
                out << geochem::BasicTranspiler::transpile(k, it->second) << "\n";
            } else {
                // Fallback to cpp file
                std::string rate_file = kinetics_dir + "/" + k + ".cpp";
                if (!std::filesystem::exists(rate_file)) {
                    throw std::runtime_error("Kinetic rate script not found in database and fallback file missing: " + rate_file);
                }
                out << "#include \"" << std::filesystem::absolute(rate_file).string() << "\"\n";
            }
        }
        
        // Generate the constexpr lookup functions for string parsing
        out << "namespace geochem {\n"
            << "namespace jit {\n"
            << "    constexpr bool str_eq(const char* a, const char* b) {\n"
            << "        while (*a && *b) { if (*a != *b) return false; a++; b++; }\n"
            << "        return *a == *b;\n"
            << "    }\n\n";

        auto write_getter = [&](const std::string& func_name, const std::vector<std::string>& names) {
            out << "    inline int " << func_name << "(const char* name) {\n";
            for (size_t i = 0; i < names.size(); ++i) {
                out << "        if (str_eq(name, \"" << names[i] << "\")) return " << i << ";\n";
            }
            out << "        return -1;\n"
                << "    }\n";
        };

        write_getter("get_phase_idx", matrices.all_phase_names);
        write_getter("get_species_idx", matrices.species_names);
        write_getter("get_element_idx", matrices.elements);
        write_getter("get_kinetics_idx", matrices.kinetics_names);

        out << "} // namespace jit\n"
            << "} // namespace geochem\n\n";

        // Include the actual backend source code
        out << "#include \"" << std::filesystem::absolute(src_dir + "/solver_backend.cpp").string() << "\"\n\n";
        out << "extern \"C\" geochem::IBackendSolver* create_solver(sycl::queue q, const geochem::SystemMatrices& m) {\n";
        for (const auto& k : matrices.kinetics_names) {
            out << "    geochem::RateRegistry::instance().register_rate(\"" << k << "\", geochem::rate_" << k << "<double>);\n";
            out << "    hipsycl::glue::reflection::enable_function_symbol_reflection(geochem::rate_" << k << "<double>);\n";
        }
        out << "    return new geochem::BackendSolver(q, m);\n";
        out << "}\n";
        out.close();

        // Prepare the compile command
        std::string cmd = "syclcc -O3 -shared -fPIC -I" + include_dir + " --acpp-targets=generic --acpp-export-all " + cpp_path + " -o " + so_path;
        
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            throw std::runtime_error("JIT Compilation failed for " + cpp_path);
        }
        std::cout << "[JIT] Compilation successful.\n";
    } else {
        std::cout << "[JIT] Loaded from cache: " << so_path << "\n";
    }

    // Load the generated dynamic library
    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        throw std::runtime_error(std::string("Failed to load JIT module: ") + dlerror());
    }

    typedef IBackendSolver* (*CreateSolverFn)(sycl::queue, const SystemMatrices&);
    CreateSolverFn create_func = (CreateSolverFn)dlsym(handle, "create_solver");
    if (!create_func) {
        throw std::runtime_error("Failed to find create_solver in JIT module");
    }

    return create_func(q, matrices);
}

} // namespace geochem
