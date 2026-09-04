#include <sycl/sycl.hpp>
#include "rates.hpp"

namespace geochem {
template<typename Real>
SYCL_EXTERNAL inline Real rate_Calcite(const geochem::RateContext<Real>& ctx) {
    Real mech_c = 0.0, mech_a = 0.0, logK25 = 0.0, e = 0.0, deltaT = 0.0, R = 0.0, Ea = 0.0, moles = 0.0, rate = 0.0;
    L10: moles=0;
    L20: if ((ctx.m()<=0) && (ctx.si("Calcite")<0)) { goto L200; }
    L30: R=8.314462;
    L40: deltaT=1/ctx.tk()-1/298.15;
    L50: e=2.718282;
    L60: Ea=14400;
    L70: logK25=-0.3;
    L90: mech_a=(sycl::pow(static_cast<Real>(10), static_cast<Real>(logK25)))*(sycl::exp(static_cast<Real>((-Ea/R*deltaT))))*ctx.act("H+");
    L100: Ea=23500;
    L110: logK25=-5.81;
    L120: mech_c=(sycl::pow(static_cast<Real>(10), static_cast<Real>(logK25)))*(sycl::exp(static_cast<Real>((-Ea/R*deltaT))));
    L130: rate=mech_a+mech_c;
    L140: if (ctx.si("Calcite")<0 && ctx.m()>0) { moles=ctx.parm(1)*rate*(1-ctx.sr("Calcite")); }
    L200: return moles*ctx.time();
    return 0.0;
}
} // namespace geochem

namespace geochem {
namespace jit {
    constexpr bool str_eq(const char* a, const char* b) {
        while (*a && *b) { if (*a != *b) return false; a++; b++; }
        return *a == *b;
    }

    inline int get_phase_idx(const char* name) {
        if (str_eq(name, "Calcite")) return 0;
        if (str_eq(name, "Aragonite")) return 1;
        if (str_eq(name, "H2O(g)")) return 2;
        if (str_eq(name, "Gypsum")) return 3;
        return -1;
    }
    inline int get_species_idx(const char* name) {
        if (str_eq(name, "CaHCO3+")) return 0;
        if (str_eq(name, "CaSO4")) return 1;
        if (str_eq(name, "CO2")) return 2;
        if (str_eq(name, "CO3-2")) return 3;
        if (str_eq(name, "HCO3-")) return 4;
        if (str_eq(name, "OH-")) return 5;
        if (str_eq(name, "CaCO3")) return 6;
        if (str_eq(name, "CaOH+")) return 7;
        if (str_eq(name, "HSO4-")) return 8;
        if (str_eq(name, "Ca+2")) return 9;
        if (str_eq(name, "H+")) return 10;
        if (str_eq(name, "SO4-2")) return 11;
        return -1;
    }
    inline int get_element_idx(const char* name) {
        if (str_eq(name, "Ca+2")) return 0;
        if (str_eq(name, "CO3-2")) return 1;
        if (str_eq(name, "SO4-2")) return 2;
        if (str_eq(name, "H+")) return 3;
        return -1;
    }
    inline int get_kinetics_idx(const char* name) {
        if (str_eq(name, "Calcite")) return 0;
        return -1;
    }
} // namespace jit
} // namespace geochem

#include "/home/max/repo/litphreeqc-MP/src/solver_backend.cpp"

extern "C" geochem::IBackendSolver* create_solver(sycl::queue q, const geochem::SystemMatrices& m) {
    geochem::RateRegistry::instance().register_rate("Calcite", geochem::rate_Calcite<double>);
    hipsycl::glue::reflection::enable_function_symbol_reflection(geochem::rate_Calcite<double>);
    return new geochem::BackendSolver(q, m);
}
