#include "rates.hpp"
#include <hipSYCL/sycl/jit.hpp>

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
    hipsycl::glue::reflection::enable_function_symbol_reflection(rate_dummy);
    hipsycl::glue::reflection::enable_function_symbol_reflection(rate_Calcite);
    hipsycl::glue::reflection::enable_function_symbol_reflection(rate_Dolomite);

    register_rate("Calcite", rate_Calcite);
    register_rate("Dolomite", rate_Dolomite);
}

void RateRegistry::register_rate(const std::string& name, RateFnPtr fn) {
    rates_[name] = fn;
}

RateFnPtr RateRegistry::get_rate(const std::string& name) const {
    auto it = rates_.find(name);
    if (it != rates_.end()) {
        return it->second;
    }
    return rate_dummy;
}

bool RateRegistry::has_rate(const std::string& name) const {
    return rates_.find(name) != rates_.end();
}

} // namespace geochem
