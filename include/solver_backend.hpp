#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include <memory>
#include "dto.hpp"
#include "device_types.hpp"
#include "rates.hpp"

namespace geochem {

struct SystemMatrices {
    int E = 0;      // Elements count
    int S = 0;      // Species count
    int P = 0;      // Mineral count
    int P_all = 0;  // All phases count
    int K = 0;      // Kinetics count
    int h_idx = -1; // H+ index in elements

    // Flattened data arrays
    std::vector<std::string> elements;
    std::vector<std::string> species_names;
    std::vector<std::string> mineral_names;
    std::vector<std::string> all_phase_names;
    std::vector<std::string> kinetics_names;
    std::unordered_map<std::string, std::string> rates_scripts;

    // Species data
    std::vector<double> species_stoich_matrix; // S x E
    std::vector<double> species_logK;          // S
    std::vector<double> species_charge;        // S
    std::vector<double> species_gamma_params;  // S x 2

    // Minerals data
    std::vector<double> mineral_stoich_matrix; // P x E
    std::vector<double> mineral_logK;          // P
    std::vector<int> mineral_to_all_idx;       // P

    // All phases data (for SI evaluation)
    std::vector<double> all_phases_stoich_matrix; // P_all x E
    std::vector<double> all_phases_logK;          // P_all

    // Kinetics data
    std::vector<double> kinetic_stoich_matrix; // K x E
    std::vector<int> kinetic_indices;          // K x 3
};

class ParallelSolverState {
public:
    int N = 0;
    int E = 0;
    int P = 0;
    int K = 0;
    sycl::queue queue;

    // USM Shared Memory Arrays (Accessible on Host and Device)
    double* temperature = nullptr;
    double* pressure = nullptr;
    double* water_mass = nullptr;

    double* element_totals = nullptr;    // E x N
    double* master_activities = nullptr; // E x N

    double* mineral_moles = nullptr;     // P x N
    double* target_si = nullptr;         // P x N
    bool* force_equality = nullptr;      // P x N

    double* kinetic_moles = nullptr;      // K x N
    double* kinetic_moles_init = nullptr; // K x N
    double* kinetic_params = nullptr;     // MAX_RATE_PARAMS x K x N

    double* charge_balance_target = nullptr; // N
    double* time_remaining = nullptr;        // N
    double* current_h = nullptr;             // N
    bool* converged = nullptr;               // N

    ParallelSolverState(int n_cells, int n_elements, int n_minerals, int n_kinetics, sycl::queue q);
    ~ParallelSolverState();

    // Disable copy, allow move
    ParallelSolverState(const ParallelSolverState&) = delete;
    ParallelSolverState& operator=(const ParallelSolverState&) = delete;
    ParallelSolverState(ParallelSolverState&& other) noexcept;
    ParallelSolverState& operator=(ParallelSolverState&& other) noexcept;
};

class IBackendSolver {
public:
    virtual ~IBackendSolver() = default;
    virtual void initialize_grid(const SystemInput& input) = 0;
    virtual SystemOutput run_simulation(double dt) = 0;
    virtual bool solve_initial_equilibration(int cell_idx, bool fix_pH, double cb_target) = 0;
    virtual ParallelSolverState& get_state() = 0;
    virtual const SystemMatrices& get_matrices() const = 0;
};

class BackendSolver : public IBackendSolver {
public:
    BackendSolver(sycl::queue q, const SystemMatrices& matrices);
    ~BackendSolver() override;

    void initialize_grid(const SystemInput& input) override;
    SystemOutput run_simulation(double dt) override;

    // Solve equilibrium for a single cell (useful for initial equilibration)
    bool solve_initial_equilibration(int cell_idx, bool fix_pH, double cb_target) override;

    ParallelSolverState& get_state() override { return *state_; }
    const SystemMatrices& get_matrices() const override { return matrices_; }

private:
    sycl::queue queue_;
    SystemMatrices matrices_;
    std::unique_ptr<ParallelSolverState> state_;

    // Device-resident constant data buffers
    double* dev_species_stoich_ = nullptr;
    double* dev_species_logK_ = nullptr;
    double* dev_species_charge_ = nullptr;
    double* dev_species_gamma_ = nullptr;

    double* dev_mineral_stoich_ = nullptr;
    double* dev_mineral_logK_ = nullptr;

    double* dev_all_phases_stoich_ = nullptr;
    double* dev_all_phases_logK_ = nullptr;

    double* dev_kinetic_stoich_ = nullptr;
    int* dev_kinetic_indices_ = nullptr;

    void allocate_device_matrices();
    void free_device_matrices();
};

bool solve_cell_equilibrium_device(
    double temp_k, double press_atm, double water_kg,
    const double* totals, double* moles_min, const double* target_si, const bool* force_eq,
    bool fix_pH, double charge_balance_target, double* ln_act,
    int E, int S, int P, int P_all, int h_idx,
    const double* species_stoich, const double* species_logK, const double* species_charge, const double* species_gamma,
    const double* mineral_stoich, const double* mineral_logK,
    const double* all_phases_stoich, const double* all_phases_logK,
    FixedVector<double, MAX_SPECIES>& final_activities,
    FixedVector<double, MAX_ALL_PHASES>& final_si,
    FixedVector<double, MAX_ALL_PHASES>& final_sr
);

} // namespace geochem
