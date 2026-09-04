#include "rates.hpp"
#include <hipSYCL/sycl/jit.hpp>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <dlfcn.h>

namespace geochem {

RateRegistry& RateRegistry::instance() {
    static RateRegistry reg;
    return reg;
}

RateRegistry::RateRegistry() {
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_0);
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_1);
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_2);
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_3);
}

void RateRegistry::register_rate(const std::string& name, RateFnPtr fn) {
    rates_[name] = fn;
}

void RateRegistry::load_kinetics_from_dir(const std::string& path) {
    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".so") {
            std::string name = entry.path().stem().string();
            std::string full_path = entry.path().string();

            void* handle = dlopen(full_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) {
                std::cerr << "Failed to load kinetics module " << full_path << ": " << dlerror() << "\n";
                continue;
            }

            // Get the rate function exported by the plugin
            typedef RateFnPtr (*GetRateFn)();
            GetRateFn get_rate_func = (GetRateFn)dlsym(handle, "get_rate_function");
            if (!get_rate_func) {
                std::cerr << "Failed to find get_rate_function in " << full_path << "\n";
                dlclose(handle);
                continue;
            }

            RateFnPtr rate_ptr = get_rate_func();
            register_rate(name, rate_ptr);
            std::cout << "Loaded dynamic kinetic rate: " << name << " from " << full_path << "\n";
        }
    }
}

RateFnPtr RateRegistry::get_rate(const std::string& name) const {
    auto it = rates_.find(name);
    if (it != rates_.end()) {
        return it->second;
    }
    throw std::runtime_error("Kinetic rate function not found for mineral: " + name);
}

bool RateRegistry::has_rate(const std::string& name) const {
    return rates_.find(name) != rates_.end();
}

} // namespace geochem
