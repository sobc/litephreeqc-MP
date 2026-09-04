#pragma once

#include <sycl/sycl.hpp>
#include <string>
#include <unordered_map>
#include "device_types.hpp"

namespace geochem {

namespace jit {
    int get_phase_idx(const char* name);
    int get_species_idx(const char* name);
    int get_element_idx(const char* name);
    int get_kinetics_idx(const char* name);
}

template<typename Real = double>
class RateContext {
    Real m_val, m0_val, tk_val, time_val;
    const Real* parms_ptr;
    const Real* act_ptr;
    const Real* si_ptr;
    const Real* sr_ptr;
    const Real* tot_ptr;
    const Real* kin_ptr;

public:
    RateContext(Real m, Real m0, Real tk, Real time,
                const Real* parms, const Real* activities,
                const Real* si, const Real* sr,
                const Real* totals, const Real* kin)
        : m_val(m), m0_val(m0), tk_val(tk), time_val(time),
          parms_ptr(parms), act_ptr(activities),
          si_ptr(si), sr_ptr(sr), tot_ptr(totals), kin_ptr(kin) {}

    Real m() const { return m_val; }
    Real m0() const { return m0_val; }
    Real tk() const { return tk_val; }
    Real tc() const { return tk_val - 273.15; }
    Real time() const { return time_val; }
    Real parm(int i) const { return parms_ptr[i - 1]; } // 1-based index

    // String resolved lookups
    Real si(const char* name) const { return si_ptr[geochem::jit::get_phase_idx(name)]; }
    Real sr(const char* name) const { return sr_ptr[geochem::jit::get_phase_idx(name)]; }
    Real act(const char* name) const { return act_ptr[geochem::jit::get_species_idx(name)]; }
    Real tot(const char* name) const { return tot_ptr[geochem::jit::get_element_idx(name)]; }
    Real kin(const char* name) const { return kin_ptr[geochem::jit::get_kinetics_idx(name)]; }

    template<typename T = void> Real gas(const char* name) const { static_assert(sizeof(T) == 0, "GAS() is not supported yet."); return 0.0; }
    template<typename T = void> Real sys(const char* name) const { static_assert(sizeof(T) == 0, "SYS() is not supported yet."); return 0.0; }
    template<typename T = void> Real mol(const char* name) const { static_assert(sizeof(T) == 0, "MOL() is not supported yet (use ACT)."); return 0.0; }
    template<typename T = void> Real la(const char* name) const { static_assert(sizeof(T) == 0, "LA() is not supported yet (use ACT)."); return 0.0; }
};

using RateFnPtr = double (*)(const RateContext<double>& ctx);

// Abstract dynamic function targets for AdaptiveCpp SSCP JIT
SYCL_EXTERNAL inline double dynamic_rate_0(const RateContext<double>& ctx) { return 0.0; }
SYCL_EXTERNAL inline double dynamic_rate_1(const RateContext<double>& ctx) { return 0.0; }
SYCL_EXTERNAL inline double dynamic_rate_2(const RateContext<double>& ctx) { return 0.0; }
SYCL_EXTERNAL inline double dynamic_rate_3(const RateContext<double>& ctx) { return 0.0; }

// Concrete pre-compiled C++ rate functions have been moved to dynamic external libraries.


// Rate Registry for host-side dispatch configuration
class RateRegistry {
public:
    static RateRegistry& instance();

    void register_rate(const std::string& name, RateFnPtr fn);
    void load_kinetics_from_dir(const std::string& path);
    RateFnPtr get_rate(const std::string& name) const;
    bool has_rate(const std::string& name) const;

private:
    RateRegistry();
    std::unordered_map<std::string, RateFnPtr> rates_;
};

} // namespace geochem
