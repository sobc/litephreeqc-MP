#include "solver_backend.hpp"
#include <hipSYCL/sycl/jit.hpp>
#include <chrono>
#include <iostream>

namespace geochem {

using namespace hipsycl::sycl::AdaptiveCpp_jit;

ParallelSolverState::ParallelSolverState(int n_cells, int n_elements, int n_minerals, int n_kinetics, sycl::queue q)
    : N(n_cells), E(n_elements), P(n_minerals), K(n_kinetics), queue(q) {
    
    temperature = sycl::malloc_shared<double>(N, queue);
    pressure = sycl::malloc_shared<double>(N, queue);
    water_mass = sycl::malloc_shared<double>(N, queue);

    element_totals = sycl::malloc_shared<double>(E * N, queue);
    master_activities = sycl::malloc_shared<double>(E * N, queue);

    mineral_moles = sycl::malloc_shared<double>(P * N, queue);
    target_si = sycl::malloc_shared<double>(P * N, queue);
    force_equality = sycl::malloc_shared<bool>(P * N, queue);

    kinetic_moles = sycl::malloc_shared<double>(K * N, queue);
    kinetic_moles_init = sycl::malloc_shared<double>(K * N, queue);
    kinetic_params = sycl::malloc_shared<double>(MAX_RATE_PARAMS * K * N, queue);

    charge_balance_target = sycl::malloc_shared<double>(N, queue);
    time_remaining = sycl::malloc_shared<double>(N, queue);
    current_h = sycl::malloc_shared<double>(N, queue);
    converged = sycl::malloc_shared<bool>(N, queue);

    // Default initialization
    for (int i = 0; i < N; ++i) {
        temperature[i] = 298.15;
        pressure[i] = 1.0;
        water_mass[i] = 1.0;
        charge_balance_target[i] = 0.0;
        time_remaining[i] = 0.0;
        current_h[i] = 10.0;
        converged[i] = true;
    }
}

ParallelSolverState::~ParallelSolverState() {
    if (temperature) sycl::free(temperature, queue);
    if (pressure) sycl::free(pressure, queue);
    if (water_mass) sycl::free(water_mass, queue);

    if (element_totals) sycl::free(element_totals, queue);
    if (master_activities) sycl::free(master_activities, queue);

    if (mineral_moles) sycl::free(mineral_moles, queue);
    if (target_si) sycl::free(target_si, queue);
    if (force_equality) sycl::free(force_equality, queue);

    if (kinetic_moles) sycl::free(kinetic_moles, queue);
    if (kinetic_moles_init) sycl::free(kinetic_moles_init, queue);
    if (kinetic_params) sycl::free(kinetic_params, queue);

    if (charge_balance_target) sycl::free(charge_balance_target, queue);
    if (time_remaining) sycl::free(time_remaining, queue);
    if (current_h) sycl::free(current_h, queue);
    if (converged) sycl::free(converged, queue);
}

ParallelSolverState::ParallelSolverState(ParallelSolverState&& other) noexcept
    : N(other.N), E(other.E), P(other.P), K(other.K), queue(other.queue),
      temperature(other.temperature), pressure(other.pressure), water_mass(other.water_mass),
      element_totals(other.element_totals), master_activities(other.master_activities),
      mineral_moles(other.mineral_moles), target_si(other.target_si), force_equality(other.force_equality),
      kinetic_moles(other.kinetic_moles), kinetic_moles_init(other.kinetic_moles_init),
      kinetic_params(other.kinetic_params), charge_balance_target(other.charge_balance_target),
      time_remaining(other.time_remaining), current_h(other.current_h), converged(other.converged) {
    
    other.temperature = nullptr;
    other.pressure = nullptr;
    other.water_mass = nullptr;
    other.element_totals = nullptr;
    other.master_activities = nullptr;
    other.mineral_moles = nullptr;
    other.target_si = nullptr;
    other.force_equality = nullptr;
    other.kinetic_moles = nullptr;
    other.kinetic_moles_init = nullptr;
    other.kinetic_params = nullptr;
    other.charge_balance_target = nullptr;
    other.time_remaining = nullptr;
    other.current_h = nullptr;
    other.converged = nullptr;
}

ParallelSolverState& ParallelSolverState::operator=(ParallelSolverState&& other) noexcept {
    if (this != &other) {
        this->~ParallelSolverState();
        N = other.N; E = other.E; P = other.P; K = other.K; queue = other.queue;
        temperature = other.temperature; pressure = other.pressure; water_mass = other.water_mass;
        element_totals = other.element_totals; master_activities = other.master_activities;
        mineral_moles = other.mineral_moles; target_si = other.target_si; force_equality = other.force_equality;
        kinetic_moles = other.kinetic_moles; kinetic_moles_init = other.kinetic_moles_init;
        kinetic_params = other.kinetic_params; charge_balance_target = other.charge_balance_target;
        time_remaining = other.time_remaining; current_h = other.current_h; converged = other.converged;

        other.temperature = nullptr; other.pressure = nullptr; other.water_mass = nullptr;
        other.element_totals = nullptr; other.master_activities = nullptr;
        other.mineral_moles = nullptr; other.target_si = nullptr; other.force_equality = nullptr;
        other.kinetic_moles = nullptr; other.kinetic_moles_init = nullptr;
        other.kinetic_params = nullptr; other.charge_balance_target = nullptr;
        other.time_remaining = nullptr; other.current_h = nullptr; other.converged = nullptr;
    }
    return *this;
}

// Device speciation solver
bool solve_cell_equilibrium_device(
    double temp_k, double press_atm, double water_kg,
    const double* totals,
    double* moles_min,
    const double* target_si,
    const bool* force_eq,
    bool fix_pH,
    double charge_balance_target,
    double* ln_act,
    int E, int S, int P, int P_all, int h_idx,
    const double* species_stoich,
    const double* species_logK,
    const double* species_charge,
    const double* species_gamma,
    const double* mineral_stoich,
    const double* mineral_logK,
    const double* all_phases_stoich,
    const double* all_phases_logK,
    FixedVector<double, MAX_SPECIES>& final_activities,
    FixedVector<double, MAX_ALL_PHASES>& final_si,
    FixedVector<double, MAX_ALL_PHASES>& final_sr
) {
    constexpr int MAX_ITER = 250;
    constexpr double TOL = 1e-11;
    constexpr double LN10 = 2.302585092994045684;

    FixedVector<double, MAX_SPECIES> molalities;
    FixedVector<double, MAX_SPECIES> gammas;
    gammas.fill(1.0);

    bool conv = false;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Compute ionic strength and activity coefficients
        double I = 0.01;
        for (int step = 0; step < 4; ++step) {
            for (int i = 0; i < S; ++i) {
                double ln_m = LN10 * species_logK[i];
                for (int e = 0; e < E; ++e) {
                    double coeff = species_stoich[i * E + e];
                    if (coeff != 0.0) {
                        ln_m += coeff * ln_act[e];
                    }
                }
                if (ln_m > 50.0) ln_m = 50.0;
                else if (ln_m < -50.0) ln_m = -50.0;

                double m = sycl::exp(ln_m);
                gammas[i] = debye_huckel_gamma_device(species_charge[i], I, species_gamma[i * 2 + 0], species_gamma[i * 2 + 1]);
                molalities[i] = m / gammas[i];
            }
            double sum_z2_m = 0.0;
            for (int i = 0; i < S; ++i) {
                sum_z2_m += (species_charge[i] * species_charge[i]) * molalities[i];
            }
            I = 0.5 * sum_z2_m;
            if (I < 1e-5) I = 1e-5;
            else if (I > 10.0) I = 10.0;
        }

        // Build Residuals
        FixedVector<double, MAX_SYS_DIM> R;
        R.fill(0.0);

        // 1. Mass balance
        for (int e = 0; e < E; ++e) {
            if (e == h_idx) {
                if (fix_pH) {
                    R[e] = 0.0;
                } else {
                    double tot_charge = 0.0;
                    for (int i = 0; i < S; ++i) {
                        tot_charge += species_charge[i] * molalities[i];
                    }
                    R[e] = tot_charge - charge_balance_target;
                }
            } else {
                double tot = 0.0;
                for (int i = 0; i < S; ++i) {
                    tot += species_stoich[i * E + e] * molalities[i];
                }
                for (int p = 0; p < P; ++p) {
                    tot += mineral_stoich[p * E + e] * moles_min[p];
                }
                R[e] = tot - totals[e];
            }
        }

        // 2. Mineral solubility constraints
        bool active_minerals[MAX_MINERALS];
        for (int p = 0; p < P; ++p) {
            double log10_iap = 0.0;
            for (int e = 0; e < E; ++e) {
                double coeff = mineral_stoich[p * E + e];
                if (coeff != 0.0) {
                    log10_iap += coeff * (ln_act[e] / LN10);
                }
            }
            double si_calc = log10_iap - mineral_logK[p];
            if (moles_min[p] <= 0.0 && si_calc < target_si[p] && !force_eq[p]) {
                R[E + p] = 0.0;
                active_minerals[p] = false;
            } else {
                R[E + p] = 1e-3 * (si_calc - target_si[p]);
                active_minerals[p] = true;
            }
        }

        if (norm_device(E + P, R) < TOL) {
            conv = true;
            break;
        }

        // 3. Construct Jacobian
        FixedMatrix<double, MAX_SYS_DIM, MAX_SYS_DIM> J;
        J.fill(0.0);

        for (int j = 0; j < E; ++j) {
            for (int k = 0; k < E; ++k) {
                double tot = 0.0;
                if (j == h_idx) {
                    if (fix_pH) {
                        if (j == k) tot = 1.0;
                    } else {
                        for (int i = 0; i < S; ++i) {
                            if (species_stoich[i * E + k] != 0.0) {
                                tot += species_charge[i] * species_stoich[i * E + k] * molalities[i];
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < S; ++i) {
                        if (species_stoich[i * E + j] != 0.0 && species_stoich[i * E + k] != 0.0) {
                            tot += species_stoich[i * E + j] * species_stoich[i * E + k] * molalities[i];
                        }
                    }
                }
                J(j, k) = tot;
            }
        }

        for (int j = 0; j < E; ++j) {
            for (int p = 0; p < P; ++p) {
                if (j == h_idx) {
                    J(j, E + p) = 0.0;
                } else {
                    J(j, E + p) = mineral_stoich[p * E + j];
                }
            }
        }

        for (int p = 0; p < P; ++p) {
            if (active_minerals[p]) {
                for (int k = 0; k < E; ++k) {
                    J(E + p, k) = 1e-3 * mineral_stoich[p * E + k] / LN10;
                }
            } else {
                J(E + p, E + p) = 1.0;
            }
        }

        // 4. Solve J * delta = -R
        FixedVector<double, MAX_SYS_DIM> neg_R;
        for (int i = 0; i < E + P; ++i) {
            neg_R[i] = -R[i];
        }

        FixedVector<double, MAX_SYS_DIM> delta;
        bool success = solve_linear_system_device(E + P, J, neg_R, delta);
        if (!success) {
            // Regularize diagonal and retry
            for (int i = 0; i < E + P; ++i) {
                J(i, i) += 1e-8;
            }
            success = solve_linear_system_device(E + P, J, neg_R, delta);
            if (!success) {
                break;
            }
        }

        // 5. Apply relaxation updates
        constexpr double MAX_D_LN = 2.302;
        for (int e = 0; e < E; ++e) {
            double d = delta[e];
            if (d > MAX_D_LN) d = MAX_D_LN;
            else if (d < -MAX_D_LN) d = -MAX_D_LN;
            ln_act[e] += d;
        }

        for (int p = 0; p < P; ++p) {
            if (active_minerals[p]) {
                moles_min[p] += delta[E + p];
                if (moles_min[p] < 0.0) moles_min[p] = 0.0;
            } else {
                moles_min[p] = 0.0;
            }
        }
    }

    // Compute final activities and saturation indices
    for (int i = 0; i < S; ++i) {
        double ln_m = 0.0;
        for (int e = 0; e < E; ++e) {
            ln_m += species_stoich[i * E + e] * ln_act[e];
        }
        final_activities[i] = sycl::exp(ln_m) * gammas[i];
    }

    constexpr double LN10_VAL = 2.302585092994045684;
    for (int p = 0; p < P_all; ++p) {
        double log10_iap = 0.0;
        for (int e = 0; e < E; ++e) {
            double coeff = all_phases_stoich[p * E + e];
            if (coeff != 0.0) {
                log10_iap += coeff * (ln_act[e] / LN10_VAL);
            }
        }
        final_si[p] = log10_iap - all_phases_logK[p];
        final_sr[p] = sycl::pow(10.0, final_si[p]);
    }

    return conv;
}

// Single-cell sub-step kinetics integrator
inline bool integrate_kinetics_single_cell_device(
    int cell_idx, int N, double temp_k, double press_atm, double water_kg,
    double* totals, double* moles_min, const double* target_si, const bool* force_eq,
    double* kinetic_moles, const double* kinetic_moles_init, const double* kinetic_params,
    double charge_balance_target, double* ln_act, double& h,
    int E, int S, int P, int P_all, int K, int h_idx,
    const double* species_stoich, const double* species_logK, const double* species_charge, const double* species_gamma,
    const double* mineral_stoich, const double* mineral_logK,
    const double* all_phases_stoich, const double* all_phases_logK,
    const double* kinetic_stoich, const int* kinetic_indices
) {
    FixedVector<double, MAX_SPECIES> activities;
    FixedVector<double, MAX_ALL_PHASES> si;
    FixedVector<double, MAX_ALL_PHASES> sr;

    // 1. Equilibrate before rate evaluation
    bool conv = solve_cell_equilibrium_device(
        temp_k, press_atm, water_kg, totals, moles_min, target_si, force_eq,
        false, charge_balance_target, ln_act,
        E, S, P, P_all, h_idx,
        species_stoich, species_logK, species_charge, species_gamma,
        mineral_stoich, mineral_logK,
        all_phases_stoich, all_phases_logK,
        activities, si, sr
    );

    if (!conv) {
        return false;
    }

    // 2. Evaluate rates via dynamic functions (specialized inline by AdaptiveCpp JIT)
    FixedVector<double, MAX_KINETICS> rate_moles;
    rate_moles.fill(0.0);

    for (int k = 0; k < K; ++k) {
        const double* parms_k = &kinetic_params[(0 * K + k) * N + cell_idx];
        const int* indices_k = &kinetic_indices[k * 3];
        double r = 0.0;

        if (k == 0) {
            r = dynamic_rate_0(kinetic_moles[k], kinetic_moles_init[k], temp_k, h,
                               parms_k, activities.data, si.data, sr.data, indices_k);
        } else if (k == 1) {
            r = dynamic_rate_1(kinetic_moles[k], kinetic_moles_init[k], temp_k, h,
                               parms_k, activities.data, si.data, sr.data, indices_k);
        } else if (k == 2) {
            r = dynamic_rate_2(kinetic_moles[k], kinetic_moles_init[k], temp_k, h,
                               parms_k, activities.data, si.data, sr.data, indices_k);
        } else if (k == 3) {
            r = dynamic_rate_3(kinetic_moles[k], kinetic_moles_init[k], temp_k, h,
                               parms_k, activities.data, si.data, sr.data, indices_k);
        }
        rate_moles[k] = r;
    }

    // 3. Trial totals
    FixedVector<double, MAX_ELEMENTS> trial_totals;
    for (int e = 0; e < E; ++e) {
        trial_totals[e] = totals[e];
        for (int k = 0; k < K; ++k) {
            double coeff = kinetic_stoich[k * E + e];
            if (coeff != 0.0) {
                trial_totals[e] += rate_moles[k] * coeff;
            }
        }
    }

    // Trial minerals and ln_act copies
    FixedVector<double, MAX_MINERALS> trial_moles_min;
    for (int p = 0; p < P; ++p) trial_moles_min[p] = moles_min[p];

    FixedVector<double, MAX_ELEMENTS> trial_ln_act;
    for (int e = 0; e < E; ++e) trial_ln_act[e] = ln_act[e];

    // 4. Equilibrate trial totals
    bool conv_trial = solve_cell_equilibrium_device(
        temp_k, press_atm, water_kg, trial_totals.data, trial_moles_min.data, target_si, force_eq,
        false, charge_balance_target, trial_ln_act.data,
        E, S, P, P_all, h_idx,
        species_stoich, species_logK, species_charge, species_gamma,
        mineral_stoich, mineral_logK,
        all_phases_stoich, all_phases_logK,
        activities, si, sr
    );

    if (conv_trial) {
        // Accept step
        for (int e = 0; e < E; ++e) {
            totals[e] = trial_totals[e];
            ln_act[e] = trial_ln_act[e];
        }
        for (int p = 0; p < P; ++p) {
            moles_min[p] = trial_moles_min[p];
        }
        for (int k = 0; k < K; ++k) {
            kinetic_moles[k] -= rate_moles[k];
            if (kinetic_moles[k] < 0.0) kinetic_moles[k] = 0.0;
        }
        return true;
    } else {
        return false;
    }
}

BackendSolver::BackendSolver(sycl::queue q, const SystemMatrices& matrices)
    : queue_(q), matrices_(matrices) {
    allocate_device_matrices();
}

BackendSolver::~BackendSolver() {
    free_device_matrices();
}

void BackendSolver::allocate_device_matrices() {
    int E = matrices_.E;
    int S = matrices_.S;
    int P = matrices_.P;
    int P_all = matrices_.P_all;
    int K = matrices_.K;

    dev_species_stoich_ = sycl::malloc_shared<double>(S * E, queue_);
    dev_species_logK_ = sycl::malloc_shared<double>(S, queue_);
    dev_species_charge_ = sycl::malloc_shared<double>(S, queue_);
    dev_species_gamma_ = sycl::malloc_shared<double>(S * 2, queue_);

    dev_mineral_stoich_ = sycl::malloc_shared<double>(P * E, queue_);
    dev_mineral_logK_ = sycl::malloc_shared<double>(P, queue_);

    dev_all_phases_stoich_ = sycl::malloc_shared<double>(P_all * E, queue_);
    dev_all_phases_logK_ = sycl::malloc_shared<double>(P_all, queue_);

    dev_kinetic_stoich_ = sycl::malloc_shared<double>(K * E, queue_);
    dev_kinetic_indices_ = sycl::malloc_shared<int>(K * 3, queue_);

    // Copy host matrices to shared memory
    for (size_t i = 0; i < matrices_.species_stoich_matrix.size(); ++i) dev_species_stoich_[i] = matrices_.species_stoich_matrix[i];
    for (size_t i = 0; i < matrices_.species_logK.size(); ++i) dev_species_logK_[i] = matrices_.species_logK[i];
    for (size_t i = 0; i < matrices_.species_charge.size(); ++i) dev_species_charge_[i] = matrices_.species_charge[i];
    for (size_t i = 0; i < matrices_.species_gamma_params.size(); ++i) dev_species_gamma_[i] = matrices_.species_gamma_params[i];

    for (size_t i = 0; i < matrices_.mineral_stoich_matrix.size(); ++i) dev_mineral_stoich_[i] = matrices_.mineral_stoich_matrix[i];
    for (size_t i = 0; i < matrices_.mineral_logK.size(); ++i) dev_mineral_logK_[i] = matrices_.mineral_logK[i];

    for (size_t i = 0; i < matrices_.all_phases_stoich_matrix.size(); ++i) dev_all_phases_stoich_[i] = matrices_.all_phases_stoich_matrix[i];
    for (size_t i = 0; i < matrices_.all_phases_logK.size(); ++i) dev_all_phases_logK_[i] = matrices_.all_phases_logK[i];

    for (size_t i = 0; i < matrices_.kinetic_stoich_matrix.size(); ++i) dev_kinetic_stoich_[i] = matrices_.kinetic_stoich_matrix[i];
    for (size_t i = 0; i < matrices_.kinetic_indices.size(); ++i) dev_kinetic_indices_[i] = matrices_.kinetic_indices[i];
}

void BackendSolver::free_device_matrices() {
    if (dev_species_stoich_) sycl::free(dev_species_stoich_, queue_);
    if (dev_species_logK_) sycl::free(dev_species_logK_, queue_);
    if (dev_species_charge_) sycl::free(dev_species_charge_, queue_);
    if (dev_species_gamma_) sycl::free(dev_species_gamma_, queue_);

    if (dev_mineral_stoich_) sycl::free(dev_mineral_stoich_, queue_);
    if (dev_mineral_logK_) sycl::free(dev_mineral_logK_, queue_);

    if (dev_all_phases_stoich_) sycl::free(dev_all_phases_stoich_, queue_);
    if (dev_all_phases_logK_) sycl::free(dev_all_phases_logK_, queue_);

    if (dev_kinetic_stoich_) sycl::free(dev_kinetic_stoich_, queue_);
    if (dev_kinetic_indices_) sycl::free(dev_kinetic_indices_, queue_);
}

void BackendSolver::initialize_grid(const SystemInput& input) {
    int N = static_cast<int>(input.cells.size());
    int E = matrices_.E;
    int P = matrices_.P;
    int K = matrices_.K;

    state_ = std::make_unique<ParallelSolverState>(N, E, P, K, queue_);

    for (int idx = 0; idx < N; ++idx) {
        const auto& cell = input.cells[idx];
        state_->temperature[idx] = cell.temperature_c + 273.15;
        state_->pressure[idx] = cell.pressure_atm;
        state_->water_mass[idx] = cell.water_mass_kg;
        state_->charge_balance_target[idx] = cell.charge_balance_target;

        // Minerals
        for (int p = 0; p < P; ++p) {
            const std::string& m_name = matrices_.mineral_names[p];
            double moles = 0.0;
            double target_si = 0.0;
            bool force_eq = false;
            for (const auto& min_input : cell.minerals) {
                if (min_input.name == m_name) {
                    moles = min_input.moles;
                    target_si = min_input.target_si;
                    force_eq = min_input.force_equality;
                    break;
                }
            }
            state_->mineral_moles[p * N + idx] = moles;
            state_->target_si[p * N + idx] = target_si;
            state_->force_equality[p * N + idx] = force_eq;
        }

        // Element totals
        for (int e = 0; e < E; ++e) {
            const std::string& e_name = matrices_.elements[e];
            auto it = cell.element_totals.find(e_name);
            double tot = (it != cell.element_totals.end()) ? it->second : 0.0;
            state_->element_totals[e * N + idx] = tot;

            // Initial activity guess
            if (e_name == "H+") {
                state_->master_activities[e * N + idx] = -cell.initial_ph * std::log(10.0);
            } else {
                state_->master_activities[e * N + idx] = -3.0 * std::log(10.0);
            }
        }

        // Kinetics
        for (int k = 0; k < K; ++k) {
            const std::string& k_name = matrices_.kinetics_names[k];
            double m = 0.0;
            double m0 = 0.0;
            std::vector<double> parms;
            for (const auto& kin_input : cell.kinetics) {
                if (kin_input.rate_name == k_name) {
                    m = kin_input.m;
                    m0 = kin_input.m0;
                    parms = kin_input.parms;
                    break;
                }
            }
            state_->kinetic_moles[k * N + idx] = m;
            state_->kinetic_moles_init[k * N + idx] = m0;
            for (size_t pi = 0; pi < parms.size() && pi < MAX_RATE_PARAMS; ++pi) {
                state_->kinetic_params[(pi * K + k) * N + idx] = parms[pi];
            }
        }
    }
}

bool BackendSolver::solve_initial_equilibration(int cell_idx, bool fix_pH, double cb_target) {
    int N = state_->N;
    int E = matrices_.E;
    int S = matrices_.S;
    int P = matrices_.P;
    int P_all = matrices_.P_all;
    int h_idx = matrices_.h_idx;

    FixedVector<double, MAX_ELEMENTS> totals;
    for (int e = 0; e < E; ++e) totals[e] = state_->element_totals[e * N + cell_idx];

    FixedVector<double, MAX_MINERALS> min_moles;
    FixedVector<double, MAX_MINERALS> target_si;
    bool force_eq[MAX_MINERALS];
    for (int p = 0; p < P; ++p) {
        min_moles[p] = state_->mineral_moles[p * N + cell_idx];
        target_si[p] = state_->target_si[p * N + cell_idx];
        force_eq[p] = state_->force_equality[p * N + cell_idx];
    }

    FixedVector<double, MAX_ELEMENTS> ln_act;
    for (int e = 0; e < E; ++e) ln_act[e] = state_->master_activities[e * N + cell_idx];

    FixedVector<double, MAX_SPECIES> activities;
    FixedVector<double, MAX_ALL_PHASES> si;
    FixedVector<double, MAX_ALL_PHASES> sr;

    bool conv = solve_cell_equilibrium_device(
        state_->temperature[cell_idx], state_->pressure[cell_idx], state_->water_mass[cell_idx],
        totals.data, min_moles.data, target_si.data, force_eq,
        fix_pH, cb_target, ln_act.data,
        E, S, P, P_all, h_idx,
        dev_species_stoich_, dev_species_logK_, dev_species_charge_, dev_species_gamma_,
        dev_mineral_stoich_, dev_mineral_logK_,
        dev_all_phases_stoich_, dev_all_phases_logK_,
        activities, si, sr
    );

    if (conv) {
        for (int e = 0; e < E; ++e) state_->master_activities[e * N + cell_idx] = ln_act[e];
        for (int p = 0; p < P; ++p) state_->mineral_moles[p * N + cell_idx] = min_moles[p];
    }
    return conv;
}

SystemOutput BackendSolver::run_simulation(double dt) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int N = state_->N;
    int E = matrices_.E;
    int S = matrices_.S;
    int P = matrices_.P;
    int P_all = matrices_.P_all;
    int K = matrices_.K;
    int h_idx = matrices_.h_idx;

    // Set up AdaptiveCpp JIT dynamic functions
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_0);
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_1);
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_2);
    hipsycl::glue::reflection::enable_function_symbol_reflection(dynamic_rate_3);
    hipsycl::glue::reflection::enable_function_symbol_reflection(rate_dummy);
    hipsycl::glue::reflection::enable_function_symbol_reflection(rate_Calcite);
    hipsycl::glue::reflection::enable_function_symbol_reflection(rate_Dolomite);

    dynamic_function_config config;
    auto& reg = RateRegistry::instance();

    if (K > 0 && matrices_.kinetics_names.size() > 0) {
        RateFnPtr fn0 = reg.get_rate(matrices_.kinetics_names[0]);
        config.define(dynamic_rate_0, fn0);
    }
    if (K > 1 && matrices_.kinetics_names.size() > 1) {
        RateFnPtr fn1 = reg.get_rate(matrices_.kinetics_names[1]);
        config.define(dynamic_rate_1, fn1);
    }
    if (K > 2 && matrices_.kinetics_names.size() > 2) {
        RateFnPtr fn2 = reg.get_rate(matrices_.kinetics_names[2]);
        config.define(dynamic_rate_2, fn2);
    }
    if (K > 3 && matrices_.kinetics_names.size() > 3) {
        RateFnPtr fn3 = reg.get_rate(matrices_.kinetics_names[3]);
        config.define(dynamic_rate_3, fn3);
    }

    // Pointers for kernel capture
    double* temp = state_->temperature;
    double* press = state_->pressure;
    double* water = state_->water_mass;
    double* totals = state_->element_totals;
    double* master_act = state_->master_activities;
    double* min_moles = state_->mineral_moles;
    double* targ_si = state_->target_si;
    bool* f_eq = state_->force_equality;
    double* kin_moles = state_->kinetic_moles;
    double* kin_moles_init = state_->kinetic_moles_init;
    double* kin_parms = state_->kinetic_params;
    double* cb_target = state_->charge_balance_target;
    double* t_rem = state_->time_remaining;
    double* cur_h = state_->current_h;
    bool* conv_arr = state_->converged;

    const double* sp_stoich = dev_species_stoich_;
    const double* sp_logK = dev_species_logK_;
    const double* sp_charge = dev_species_charge_;
    const double* sp_gamma = dev_species_gamma_;
    const double* m_stoich = dev_mineral_stoich_;
    const double* m_logK = dev_mineral_logK_;
    const double* all_p_stoich = dev_all_phases_stoich_;
    const double* all_p_logK = dev_all_phases_logK_;
    const double* k_stoich = dev_kinetic_stoich_;
    const int* k_indices = dev_kinetic_indices_;

    // Initialize step arrays
    for (int i = 0; i < N; ++i) {
        t_rem[i] = dt;
        cur_h[i] = dt / 10.0;
        conv_arr[i] = true;
    }

    // Apply JIT specialization configuration to kernel
    auto specialized_kernel = config.apply([=](sycl::id<1> item) {
        int idx = item[0];
        
        FixedVector<double, MAX_ELEMENTS> cell_totals;
        FixedVector<double, MAX_MINERALS> cell_min_moles;
        FixedVector<double, MAX_MINERALS> cell_target_si;
        bool cell_force_eq[MAX_MINERALS];
        FixedVector<double, MAX_KINETICS> cell_kin_moles;
        FixedVector<double, MAX_KINETICS> cell_kin_init;
        FixedVector<double, MAX_ELEMENTS> cell_ln_act;

        for (int e = 0; e < E; ++e) {
            cell_totals[e] = totals[e * N + idx];
            cell_ln_act[e] = master_act[e * N + idx];
        }
        for (int p = 0; p < P; ++p) {
            cell_min_moles[p] = min_moles[p * N + idx];
            cell_target_si[p] = targ_si[p * N + idx];
            cell_force_eq[p] = f_eq[p * N + idx];
        }
        for (int k = 0; k < K; ++k) {
            cell_kin_moles[k] = kin_moles[k * N + idx];
            cell_kin_init[k] = kin_moles_init[k * N + idx];
        }

        while (t_rem[idx] > 1e-6) {
            double h_step = cur_h[idx];
            if (h_step > t_rem[idx]) {
                h_step = t_rem[idx];
            }

            bool accepted = integrate_kinetics_single_cell_device(
                idx, N, temp[idx], press[idx], water[idx],
                cell_totals.data, cell_min_moles.data, cell_target_si.data, cell_force_eq,
                cell_kin_moles.data, cell_kin_init.data, kin_parms,
                cb_target[idx], cell_ln_act.data, h_step,
                E, S, P, P_all, K, h_idx,
                sp_stoich, sp_logK, sp_charge, sp_gamma,
                m_stoich, m_logK,
                all_p_stoich, all_p_logK,
                k_stoich, k_indices
            );

            if (accepted) {
                t_rem[idx] -= h_step;
                double next_h = h_step * 1.5;
                if (next_h > t_rem[idx]) next_h = t_rem[idx];
                cur_h[idx] = (next_h < 1e-4) ? 1e-4 : next_h;
            } else {
                cur_h[idx] = h_step / 2.0;
                if (cur_h[idx] < 1e-6) {
                    conv_arr[idx] = false;
                    break;
                }
            }
        }

        // Commit cell state back to USM arrays
        for (int e = 0; e < E; ++e) {
            totals[e * N + idx] = cell_totals[e];
            master_act[e * N + idx] = cell_ln_act[e];
        }
        for (int p = 0; p < P; ++p) {
            min_moles[p * N + idx] = cell_min_moles[p];
        }
        for (int k = 0; k < K; ++k) {
            kin_moles[k * N + idx] = cell_kin_moles[k];
        }
    });

    // Launch parallel SYCL kernel
    queue_.parallel_for(sycl::range<1>(N), specialized_kernel).wait();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Assemble SystemOutput DTO
    SystemOutput output;
    output.elapsed_seconds = elapsed;
    output.all_converged = true;
    output.cells.resize(N);

    for (int idx = 0; idx < N; ++idx) {
        CellResult& res = output.cells[idx];
        res.cell_id = idx;
        res.converged = conv_arr[idx];
        if (!res.converged) output.all_converged = false;

        res.final_ph = -master_act[h_idx * N + idx] / std::log(10.0);

        for (int e = 0; e < E; ++e) {
            res.element_totals[matrices_.elements[e]] = totals[e * N + idx];
        }
        for (int p = 0; p < P; ++p) {
            res.mineral_moles[matrices_.mineral_names[p]] = min_moles[p * N + idx];
        }
        for (int k = 0; k < K; ++k) {
            res.kinetic_moles[matrices_.kinetics_names[k]] = kin_moles[k * N + idx];
        }
    }

    return output;
}

} // namespace geochem
