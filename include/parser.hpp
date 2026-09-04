#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "dto.hpp"
#include "solver_backend.hpp"

namespace geochem {

struct ParsedSpecies {
    std::string name;
    std::unordered_map<std::string, double> reaction; // master_spec -> coeff
    double logk = 0.0;
    double gamma_a0 = 0.0;
    double gamma_b0 = 0.0;
    double charge = 0.0;
};

struct ParsedPhase {
    std::string name;
    std::unordered_map<std::string, double> reaction; // master_spec -> coeff
    double logk = 0.0;
};

class PhreeqcDatabase {
public:
    std::unordered_map<std::string, std::string> master_species; // Element -> Master species name
    std::unordered_map<std::string, double> elements_gfw;
    std::unordered_map<std::string, ParsedSpecies> species;
    std::unordered_map<std::string, ParsedPhase> phases;
    std::unordered_map<std::string, std::string> rates_scripts; // Mineral name -> Basic script

    bool parse(const std::string& filepath);

private:
    std::unordered_map<std::string, double> parse_reaction_side(const std::string& side_str);
};

class PqiParser {
public:
    double temp_c = 25.0;
    double initial_ph = 7.0;
    double steps = 100.0;

    std::vector<std::string> elements;
    std::unordered_map<std::string, double> totals;

    std::vector<std::string> minerals;
    std::unordered_map<std::string, double> mineral_si;
    std::unordered_map<std::string, double> mineral_moles;

    std::vector<KineticCompInput> kinetics;

    bool parse_pqi(const std::string& filepath, const PhreeqcDatabase& db);
};

// Builder for SystemMatrices and initial two-stage equilibration
class SystemBuilder {
public:
    static SystemMatrices build_system_matrices(const PhreeqcDatabase& db,
                                                const PqiParser& pqi);

    static SystemInput create_system_input(const PhreeqcDatabase& db,
                                          const PqiParser& pqi,
                                          const SystemMatrices& matrices,
                                          int num_cells);

    static void equilibrate_initial_state(IBackendSolver& solver,
                                          SystemInput& input);
};

} // namespace geochem
