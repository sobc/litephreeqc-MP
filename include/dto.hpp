#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace geochem {

struct KineticCompInput {
    std::string rate_name;
    double m = 0.0;
    double m0 = 0.0;
    std::vector<double> parms;
    std::string formula;
};

struct MineralCompInput {
    std::string name;
    double target_si = 0.0;
    double moles = 10.0;
    bool force_equality = false;
};

struct CellState {
    int cell_id = 0;
    double temperature_c = 25.0;
    double pressure_atm = 1.0;
    double water_mass_kg = 1.0;
    double initial_ph = 7.0;
    double charge_balance_target = 0.0;

    // Concentrations in mol/kgw
    std::unordered_map<std::string, double> element_totals;
    std::vector<MineralCompInput> minerals;
    std::vector<KineticCompInput> kinetics;
};

struct SystemInput {
    std::vector<std::string> elements;
    std::vector<std::string> mineral_names;
    std::vector<std::string> kinetic_names;
    std::vector<CellState> cells;
    double dt = 100.0;
};

struct CellResult {
    int cell_id = 0;
    double final_ph = 7.0;
    std::unordered_map<std::string, double> element_totals;
    std::unordered_map<std::string, double> mineral_moles;
    std::unordered_map<std::string, double> kinetic_moles;
    bool converged = true;
};

struct SystemOutput {
    std::vector<CellResult> cells;
    double elapsed_seconds = 0.0;
    bool all_converged = true;
};

} // namespace geochem
