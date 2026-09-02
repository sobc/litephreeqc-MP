#pragma once

#include <sycl/sycl.hpp>
#include <string>
#include <unordered_map>
#include "device_types.hpp"

namespace geochem {

using RateFnPtr = double (*)(double m, double m0, double tk, double time,
                             const double* parms,
                             const double* activities,
                             const double* si,
                             const double* sr,
                             const int* indices);

// Abstract dynamic function targets for AdaptiveCpp SSCP JIT
SYCL_EXTERNAL inline double dynamic_rate_0(double m, double m0, double tk, double time,
                                           const double* parms, const double* activities,
                                           const double* si, const double* sr, const int* indices) {
    return 0.0;
}

SYCL_EXTERNAL inline double dynamic_rate_1(double m, double m0, double tk, double time,
                                           const double* parms, const double* activities,
                                           const double* si, const double* sr, const int* indices) {
    return 0.0;
}

SYCL_EXTERNAL inline double dynamic_rate_2(double m, double m0, double tk, double time,
                                           const double* parms, const double* activities,
                                           const double* si, const double* sr, const int* indices) {
    return 0.0;
}

SYCL_EXTERNAL inline double dynamic_rate_3(double m, double m0, double tk, double time,
                                           const double* parms, const double* activities,
                                           const double* si, const double* sr, const int* indices) {
    return 0.0;
}

// Concrete pre-compiled C++ rate functions
SYCL_EXTERNAL inline double rate_dummy(double m, double m0, double tk, double time,
                                       const double* parms, const double* activities,
                                       const double* si, const double* sr, const int* indices) {
    return 0.0;
}

SYCL_EXTERNAL inline double rate_Calcite(double m, double m0, double tk, double time,
                                         const double* parms, const double* activities,
                                         const double* si, const double* sr, const int* indices) {
    double M = m;
    double TK = tk;

    double act_H = activities[indices[0]];
    double si_Calcite = si[indices[1]];
    double sr_Calcite = sr[indices[1]];

    if (M <= 0.0 && si_Calcite < 0.0) {
        return 0.0;
    }

    constexpr double R = 8.314462;
    double deltaT = 1.0 / TK - 1.0 / 298.15;
    constexpr double e = 2.718282;

    // Mechanism 1 (acid)
    constexpr double Ea_a = 14400.0;
    constexpr double logK25_a = -0.3;
    double mech_a = sycl::pow(10.0, logK25_a) * sycl::pow(e, -Ea_a / R * deltaT) * act_H;

    // Mechanism 2 (neutral / water)
    constexpr double Ea_c = 23500.0;
    constexpr double logK25_c = -5.81;
    double mech_c = sycl::pow(10.0, logK25_c) * sycl::pow(e, -Ea_c / R * deltaT);

    double rate = mech_a + mech_c;
    double moles = parms[0] * rate * (1.0 - sr_Calcite);
    return moles * time;
}

SYCL_EXTERNAL inline double rate_Dolomite(double m, double m0, double tk, double time,
                                          const double* parms, const double* activities,
                                          const double* si, const double* sr, const int* indices) {
    double M = m;
    double TK = tk;

    double act_H = activities[indices[0]];
    double si_Dolomite = si[indices[1]];
    double sr_Dolomite = sr[indices[1]];

    if (M <= 0.0 && si_Dolomite < 0.0) {
        return 0.0;
    }

    constexpr double R = 8.314462;
    double deltaT = 1.0 / TK - 1.0 / 298.15;
    constexpr double e = 2.718282;

    // Mechanism 1 (acid)
    constexpr double Ea_a = 36100.0;
    constexpr double logK25_a = -3.19;
    double mech_a = sycl::pow(10.0, logK25_a) * sycl::pow(e, -Ea_a / R * deltaT) * sycl::pow(act_H, 0.5);

    // Mechanism 2 (neutral)
    constexpr double Ea_c = 52200.0;
    constexpr double logK25_c = -7.53;
    double mech_c = sycl::pow(10.0, logK25_c) * sycl::pow(e, -Ea_c / R * deltaT);

    double rate = mech_a + mech_c;
    double moles = parms[0] * rate * (1.0 - sr_Dolomite);
    return moles * time;
}

// Rate Registry for host-side dispatch configuration
class RateRegistry {
public:
    static RateRegistry& instance();

    void register_rate(const std::string& name, RateFnPtr fn);
    RateFnPtr get_rate(const std::string& name) const;
    bool has_rate(const std::string& name) const;

private:
    RateRegistry();
    std::unordered_map<std::string, RateFnPtr> rates_;
};

} // namespace geochem
