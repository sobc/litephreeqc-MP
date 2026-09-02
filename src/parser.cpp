#include "parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <unordered_set>

namespace geochem {

static inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

static inline std::string to_upper(const std::string& str) {
    std::string res = str;
    for (char& c : res) c = std::toupper(c);
    return res;
}

static double extract_charge(const std::string& name) {
    size_t plus = name.rfind('+');
    size_t minus = name.rfind('-');
    if (plus != std::string::npos) {
        std::string num = name.substr(plus + 1);
        if (num.empty()) return 1.0;
        try { return std::stod(num); } catch (...) { return 1.0; }
    }
    if (minus != std::string::npos) {
        std::string num = name.substr(minus + 1);
        if (num.empty()) return -1.0;
        try { return -std::stod(num); } catch (...) { return -1.0; }
    }
    return 0.0;
}

std::unordered_map<std::string, double> PhreeqcDatabase::parse_reaction_side(const std::string& side_str) {
    std::unordered_map<std::string, double> stoich;
    
    // Split on ' + ' (with spaces) to prevent splitting on charges like Ca+2 or H+
    std::vector<std::string> parts;
    std::string s = side_str;
    size_t pos = 0;
    while ((pos = s.find(" + ")) != std::string::npos) {
        parts.push_back(trim(s.substr(0, pos)));
        s = s.substr(pos + 3);
    }
    parts.push_back(trim(s));

    for (const auto& p : parts) {
        if (p.empty()) continue;

        size_t idx = 0;
        while (idx < p.size() && (std::isdigit(p[idx]) || p[idx] == '.')) {
            idx++;
        }
        double coeff = 1.0;
        std::string spec = p;
        if (idx > 0 && idx < p.size()) {
            coeff = std::stod(p.substr(0, idx));
            spec = trim(p.substr(idx));
        } else if (idx == p.size()) {
            coeff = std::stod(p);
            spec = "";
        }
        if (!spec.empty()) {
            stoich[spec] += coeff;
        }
    }
    return stoich;
}
static std::string determine_defined_species(
    const std::unordered_map<std::string, double>& reactants,
    const std::unordered_map<std::string, double>& products,
    const std::unordered_set<std::string>& master_species) {

    std::vector<std::string> non_masters;
    for (const auto& p : reactants) {
        if (master_species.find(p.first) == master_species.end()) {
            non_masters.push_back(p.first);
        }
    }
    for (const auto& p : products) {
        if (master_species.find(p.first) == master_species.end()) {
            non_masters.push_back(p.first);
        }
    }

    if (non_masters.size() == 1) {
        return non_masters[0];
    } else if (reactants.size() == 1) {
        return reactants.begin()->first;
    } else {
        return products.begin()->first;
    }
}

static std::unordered_map<std::string, double> compute_formation_reaction(
    const std::string& defined_species,
    const std::unordered_map<std::string, double>& reactants,
    const std::unordered_map<std::string, double>& products) {

    std::unordered_map<std::string, double> formation;
    auto prod_it = products.find(defined_species);
    if (prod_it != products.end()) {
        double c_prod = prod_it->second;
        for (const auto& r : reactants) {
            formation[r.first] = r.second / c_prod;
        }
        for (const auto& p : products) {
            if (p.first != defined_species) {
                formation[p.first] = -p.second / c_prod;
            }
        }
    } else {
        auto react_it = reactants.find(defined_species);
        double c_react = (react_it != reactants.end()) ? react_it->second : 1.0;
        for (const auto& p : products) {
            formation[p.first] = p.second / c_react;
        }
        for (const auto& r : reactants) {
            if (r.first != defined_species) {
                formation[r.first] = -r.second / c_react;
            }
        }
    }
    return formation;
}

bool PhreeqcDatabase::parse(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open database file: " << filepath << "\n";
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Pass 1: find master species set
    std::unordered_set<std::string> master_species_set = {"H+", "e-", "H2O", "H", "O"};
    std::string current_sec = "";
    for (const auto& raw_l : lines) {
        std::string l = trim(raw_l);
        if (l.empty() || l[0] == '#') continue;
        if (l == "SOLUTION_MASTER_SPECIES" || l == "SOLUTION_SPECIES" || l == "PHASES" || l == "RATES") {
            current_sec = l;
            continue;
        }
        if (current_sec == "SOLUTION_MASTER_SPECIES") {
            std::stringstream ss(l);
            std::string elt, spec;
            if (ss >> elt >> spec) {
                master_species_set.insert(spec);
            }
        }
    }

    // Pass 2: parse all blocks
    std::string current_section = "";
    size_t i = 0;
    while (i < lines.size()) {
        std::string l = trim(lines[i]);
        if (l.empty() || l[0] == '#') {
            i++;
            continue;
        }

        if (l == "SOLUTION_MASTER_SPECIES" || l == "SOLUTION_SPECIES" ||
            l == "PHASES" || l == "RATES") {
            current_section = l;
            i++;
            continue;
        } else if (l == "END" || (std::isupper(l[0]) && l.find(' ') == std::string::npos && l.find('=') == std::string::npos && l[0] != '-' && l != "Calcite" && l != "Gypsum" && l != "Dolomite" && l.find('+') == std::string::npos && l.find('-') == std::string::npos && l.length() > 6)) {
            // Unknown keyword block
            if (l.find('_') != std::string::npos) {
                current_section = "";
                i++;
                continue;
            }
        }

        if (current_section == "SOLUTION_MASTER_SPECIES") {
            std::stringstream ss(l);
            std::string elt, spec, dummy1, dummy2, gfw_str;
            if (ss >> elt >> spec) {
                if (master_species.find(elt) == master_species.end()) {
                    master_species[elt] = spec;
                }
                if (ss >> dummy1 >> dummy2 >> gfw_str) {
                    try { elements_gfw[elt] = std::stod(gfw_str); } catch (...) {}
                }
            }
            i++;
        } else if (current_section == "SOLUTION_SPECIES") {
            std::string rxn_line = l;
            double logk = 0.0;
            double gamma_a0 = 0.0;
            double gamma_b0 = 0.0;
            i++;

            while (i < lines.size()) {
                std::string sub_line = trim(lines[i]);
                if (sub_line.empty() || sub_line[0] == '#') { i++; continue; }
                if (sub_line.find("-log_k") == 0) {
                    std::stringstream sss(sub_line);
                    std::string key; double val;
                    if (sss >> key >> val) logk = val;
                    i++;
                } else if (sub_line.find("-analytic") == 0) {
                    std::stringstream sss(sub_line);
                    std::string key;
                    double a[6] = {0.0};
                    sss >> key;
                    for (int idx = 0; idx < 6 && (sss >> a[idx]); ++idx);
                    constexpr double T = 298.15;
                    logk = a[0] + a[1]*T + a[2]/T + a[3]*std::log10(T) + a[4]/(T*T) + a[5]*(T*T);
                    i++;
                } else if (sub_line.find("-gamma") == 0) {
                    std::stringstream sss(sub_line);
                    std::string key;
                    sss >> key >> gamma_a0 >> gamma_b0;
                    i++;
                } else if (sub_line.find('=') != std::string::npos ||
                           sub_line == "SOLUTION_MASTER_SPECIES" ||
                           sub_line == "SOLUTION_SPECIES" ||
                           sub_line == "PHASES" ||
                           sub_line == "RATES") {
                    break;
                } else {
                    i++;
                }
            }

            size_t eq_pos = rxn_line.find('=');
            if (eq_pos != std::string::npos) {
                std::string left = rxn_line.substr(0, eq_pos);
                std::string right = rxn_line.substr(eq_pos + 1);
                auto react = parse_reaction_side(left);
                auto prod = parse_reaction_side(right);

                std::string defined_spec = determine_defined_species(react, prod, master_species_set);
                auto formation = compute_formation_reaction(defined_spec, react, prod);

                ParsedSpecies ps;
                ps.name = defined_spec;
                ps.reaction = formation;
                ps.logk = logk;
                ps.gamma_a0 = gamma_a0;
                ps.gamma_b0 = gamma_b0;
                ps.charge = extract_charge(defined_spec);
                species[defined_spec] = ps;
            }
        } else if (current_section == "PHASES") {
            std::string phase_name = l;
            i++;
            if (i >= lines.size()) break;
            std::string rxn_line = trim(lines[i]);
            double logk = 0.0;
            i++;

            while (i < lines.size()) {
                std::string sub_line = trim(lines[i]);
                if (sub_line.empty() || sub_line[0] == '#') { i++; continue; }
                if (sub_line.find("-log_k") == 0) {
                    std::stringstream sss(sub_line);
                    std::string key; double val;
                    if (sss >> key >> val) logk = val;
                    i++;
                } else if (sub_line.find("-analytic") == 0) {
                    std::stringstream sss(sub_line);
                    std::string key;
                    double a[6] = {0.0};
                    sss >> key;
                    for (int idx = 0; idx < 6 && (sss >> a[idx]); ++idx);
                    constexpr double T = 298.15;
                    logk = a[0] + a[1]*T + a[2]/T + a[3]*std::log10(T) + a[4]/(T*T) + a[5]*(T*T);
                    i++;
                } else if (sub_line.find('=') != std::string::npos ||
                           sub_line == "SOLUTION_MASTER_SPECIES" ||
                           sub_line == "SOLUTION_SPECIES" ||
                           sub_line == "PHASES" ||
                           sub_line == "RATES" ||
                           (sub_line[0] != '-' && sub_line.find(' ') == std::string::npos)) {
                    break;
                } else {
                    i++;
                }
            }

            size_t eq_pos = rxn_line.find('=');
            if (eq_pos != std::string::npos) {
                std::string right = rxn_line.substr(eq_pos + 1);
                auto prod = parse_reaction_side(right);

                ParsedPhase pp;
                pp.name = phase_name;
                pp.reaction = prod;
                pp.logk = logk;
                phases[phase_name] = pp;
            }
        } else {
            i++;
        }
    }
    return true;
}

bool PqiParser::parse_pqi(const std::string& filepath, const PhreeqcDatabase& db) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open PQI file: " << filepath << "\n";
        return false;
    }

    std::string line;
    std::string current_block = "";
    bool parsed_solution = false;
    bool parsed_equilibrium = false;
    bool parsed_kinetics = false;

    while (std::getline(file, line)) {
        std::string l = trim(line);
        if (l.empty() || l[0] == '#') continue;

        std::stringstream ss(l);
        std::string first_word;
        ss >> first_word;
        std::string upper_word = to_upper(first_word);

        if (upper_word.find("SOLUTION") == 0) {
            if (!parsed_solution) {
                current_block = "SOLUTION";
                parsed_solution = true;
            } else {
                current_block = "SKIP";
            }
            continue;
        } else if (upper_word.find("EQUILIBRIUM_PHASES") == 0 || upper_word.find("PURE") == 0) {
            if (!parsed_equilibrium) {
                current_block = "EQUILIBRIUM_PHASES";
                parsed_equilibrium = true;
            } else {
                current_block = "SKIP";
            }
            continue;
        } else if (upper_word.find("KINETICS") == 0) {
            if (!parsed_kinetics) {
                current_block = "KINETICS";
                parsed_kinetics = true;
            } else {
                current_block = "SKIP";
            }
            continue;
        } else if (upper_word == "SELECTED_OUTPUT" || upper_word == "USER_PUNCH" ||
                   upper_word == "PRINT" || upper_word == "END") {
            current_block = "";
            continue;
        }

        if (current_block == "SOLUTION") {
            std::string p0 = to_upper(first_word);
            if (p0 == "TEMP") {
                ss >> temp_c;
            } else if (p0 == "PH") {
                ss >> initial_ph;
                if (std::find(elements.begin(), elements.end(), "H+") == elements.end()) {
                    elements.push_back("H+");
                }
                totals["H+"] = 0.0;
            } else if (p0[0] != '-') {
                std::string elt_name = first_word;
                auto it = db.master_species.find(elt_name);
                if (it != db.master_species.end()) {
                    double conc = 0.0;
                    if (ss >> conc) {
                        std::string m_spec = it->second;
                        if (std::find(elements.begin(), elements.end(), m_spec) == elements.end()) {
                            elements.push_back(m_spec);
                        }
                        totals[m_spec] = conc;
                    }
                }
            }
        } else if (current_block == "EQUILIBRIUM_PHASES") {
            if (l[0] != '-') {
                std::string phase = first_word;
                double si = 0.0;
                double moles = 10.0;
                ss >> si >> moles;
                minerals.push_back(phase);
                mineral_si[phase] = si;
                mineral_moles[phase] = moles;
            }
        } else if (current_block == "KINETICS") {
            if (l[0] == '-') {
                std::string p0 = to_upper(first_word);
                if (p0 == "-STEPS") {
                    ss >> steps;
                }
            } else {
                KineticCompInput kin;
                kin.rate_name = first_word;
                kin.m = 0.0;
                kin.m0 = 0.0;
                
                std::streampos pos = file.tellg();
                std::string sub_l;
                while (std::getline(file, sub_l)) {
                    std::string trimmed_sub = trim(sub_l);
                    if (trimmed_sub.empty() || trimmed_sub[0] == '#') continue;
                    if (trimmed_sub[0] != '-') {
                        file.seekg(pos);
                        break;
                    }
                    pos = file.tellg();
                    std::stringstream sub_ss(trimmed_sub);
                    std::string opt;
                    sub_ss >> opt;
                    std::string upper_opt = to_upper(opt);
                    if (upper_opt == "-M0") sub_ss >> kin.m0;
                    else if (upper_opt == "-M") sub_ss >> kin.m;
                    else if (upper_opt == "-PARMS") {
                        double p_val;
                        while (sub_ss >> p_val) kin.parms.push_back(p_val);
                    } else if (upper_opt == "-STEPS") {
                        sub_ss >> steps;
                    }
                }
                kinetics.push_back(kin);
            }
        }
    }
    return true;
}

SystemMatrices SystemBuilder::build_system_matrices(const PhreeqcDatabase& db, const PqiParser& pqi) {
    SystemMatrices sm;

    // 1. Gather and deduplicate elements
    std::vector<std::string> all_elements = pqi.elements;

    for (const auto& m_name : pqi.minerals) {
        auto it = db.phases.find(m_name);
        if (it != db.phases.end()) {
            for (const auto& r : it->second.reaction) {
                if (r.first != "H2O" && r.first != "e-") {
                    all_elements.push_back(r.first);
                }
            }
        }
    }

    for (const auto& k : pqi.kinetics) {
        auto it = db.phases.find(k.rate_name);
        if (it != db.phases.end()) {
            for (const auto& r : it->second.reaction) {
                if (r.first != "H2O" && r.first != "e-") {
                    all_elements.push_back(r.first);
                }
            }
        }
    }

    for (const auto& e : all_elements) {
        if (std::find(sm.elements.begin(), sm.elements.end(), e) == sm.elements.end()) {
            sm.elements.push_back(e);
        }
    }

    // Move H+ to the end
    auto h_it = std::find(sm.elements.begin(), sm.elements.end(), "H+");
    if (h_it != sm.elements.end()) {
        sm.elements.erase(h_it);
    }
    sm.elements.push_back("H+");

    sm.E = static_cast<int>(sm.elements.size());
    sm.h_idx = sm.E - 1;

    std::unordered_map<std::string, int> elt_to_idx;
    for (int e = 0; e < sm.E; ++e) elt_to_idx[sm.elements[e]] = e;

    // 2. Filter aqueous species
    for (const auto& pair : db.species) {
        const auto& spec = pair.second;
        bool valid = true;
        for (const auto& r : spec.reaction) {
            if (r.first == "H2O") continue;
            if (elt_to_idx.find(r.first) == elt_to_idx.end()) {
                valid = false;
                break;
            }
        }
        if (valid) {
            sm.species_names.push_back(spec.name);
        }
    }

    sm.S = static_cast<int>(sm.species_names.size());
    sm.species_stoich_matrix.resize(sm.S * sm.E, 0.0);
    sm.species_logK.resize(sm.S, 0.0);
    sm.species_charge.resize(sm.S, 0.0);
    sm.species_gamma_params.resize(sm.S * 2, 0.0);

    for (int i = 0; i < sm.S; ++i) {
        const auto& spec = db.species.at(sm.species_names[i]);
        sm.species_logK[i] = spec.logk;
        sm.species_charge[i] = spec.charge;
        sm.species_gamma_params[i * 2 + 0] = spec.gamma_a0;
        sm.species_gamma_params[i * 2 + 1] = spec.gamma_b0;

        for (const auto& r : spec.reaction) {
            auto it = elt_to_idx.find(r.first);
            if (it != elt_to_idx.end()) {
                sm.species_stoich_matrix[i * sm.E + it->second] = r.second;
            }
        }
    }

    // 3. Minerals (Equilibrium Phases)
    sm.mineral_names = pqi.minerals;
    sm.P = static_cast<int>(sm.mineral_names.size());
    sm.mineral_stoich_matrix.resize(sm.P * sm.E, 0.0);
    sm.mineral_logK.resize(sm.P, 0.0);
    sm.mineral_to_all_idx.resize(sm.P, 0);

    for (int p = 0; p < sm.P; ++p) {
        const auto& m_name = sm.mineral_names[p];
        auto it = db.phases.find(m_name);
        if (it != db.phases.end()) {
            sm.mineral_logK[p] = it->second.logk;
            for (const auto& r : it->second.reaction) {
                auto e_it = elt_to_idx.find(r.first);
                if (e_it != elt_to_idx.end()) {
                    sm.mineral_stoich_matrix[p * sm.E + e_it->second] = r.second;
                }
            }
        }
    }

    // 4. All relevant phases (whose elements are a subset of system elements)
    for (const auto& pair : db.phases) {
        bool valid = true;
        for (const auto& r : pair.second.reaction) {
            if (r.first == "H2O") continue;
            if (elt_to_idx.find(r.first) == elt_to_idx.end()) {
                valid = false;
                break;
            }
        }
        if (valid) {
            sm.all_phase_names.push_back(pair.first);
        }
    }
    sm.P_all = static_cast<int>(sm.all_phase_names.size());
    sm.all_phases_stoich_matrix.resize(sm.P_all * sm.E, 0.0);
    sm.all_phases_logK.resize(sm.P_all, 0.0);

    for (int p = 0; p < sm.P_all; ++p) {
        const auto& ph = db.phases.at(sm.all_phase_names[p]);
        sm.all_phases_logK[p] = ph.logk;
        for (const auto& r : ph.reaction) {
            auto it = elt_to_idx.find(r.first);
            if (it != elt_to_idx.end()) {
                sm.all_phases_stoich_matrix[p * sm.E + it->second] = r.second;
            }
        }
    }

    for (int p = 0; p < sm.P; ++p) {
        auto it = std::find(sm.all_phase_names.begin(), sm.all_phase_names.end(), sm.mineral_names[p]);
        sm.mineral_to_all_idx[p] = (it != sm.all_phase_names.end()) ? static_cast<int>(std::distance(sm.all_phase_names.begin(), it)) : 0;
    }

    // 5. Kinetics
    for (const auto& k : pqi.kinetics) {
        sm.kinetics_names.push_back(k.rate_name);
    }
    sm.K = static_cast<int>(sm.kinetics_names.size());
    sm.kinetic_stoich_matrix.resize(sm.K * sm.E, 0.0);
    sm.kinetic_indices.resize(sm.K * 3, -1);

    int h_species_idx = -1;
    auto sp_h_it = std::find(sm.species_names.begin(), sm.species_names.end(), "H+");
    if (sp_h_it != sm.species_names.end()) {
        h_species_idx = static_cast<int>(std::distance(sm.species_names.begin(), sp_h_it));
    }

    for (int k = 0; k < sm.K; ++k) {
        const std::string& k_name = sm.kinetics_names[k];
        auto it = db.phases.find(k_name);
        if (it != db.phases.end()) {
            for (const auto& r : it->second.reaction) {
                auto e_it = elt_to_idx.find(r.first);
                if (e_it != elt_to_idx.end()) {
                    sm.kinetic_stoich_matrix[k * sm.E + e_it->second] = r.second;
                }
            }
        }

        int phase_idx = -1;
        auto p_it = std::find(sm.all_phase_names.begin(), sm.all_phase_names.end(), k_name);
        if (p_it != sm.all_phase_names.end()) {
            phase_idx = static_cast<int>(std::distance(sm.all_phase_names.begin(), p_it));
        }

        sm.kinetic_indices[k * 3 + 0] = h_species_idx;
        sm.kinetic_indices[k * 3 + 1] = phase_idx;
        sm.kinetic_indices[k * 3 + 2] = phase_idx;
    }

    return sm;
}

SystemInput SystemBuilder::create_system_input(const PhreeqcDatabase& db,
                                              const PqiParser& pqi,
                                              const SystemMatrices& matrices,
                                              int num_cells) {
    SystemInput input;
    input.elements = matrices.elements;
    input.mineral_names = matrices.mineral_names;
    input.kinetic_names = matrices.kinetics_names;
    input.dt = pqi.steps;
    input.cells.resize(num_cells);

    for (int idx = 0; idx < num_cells; ++idx) {
        CellState& cell = input.cells[idx];
        cell.cell_id = idx;
        cell.temperature_c = pqi.temp_c;
        cell.pressure_atm = 1.0;
        cell.water_mass_kg = 1.0;
        cell.initial_ph = pqi.initial_ph;
        cell.charge_balance_target = 0.0;

        // Element totals
        for (const auto& e : matrices.elements) {
            auto it = pqi.totals.find(e);
            cell.element_totals[e] = (it != pqi.totals.end()) ? it->second : 0.0;
        }

        // Minerals
        for (const auto& m_name : matrices.mineral_names) {
            MineralCompInput min_comp;
            min_comp.name = m_name;
            auto si_it = pqi.mineral_si.find(m_name);
            auto mol_it = pqi.mineral_moles.find(m_name);
            min_comp.target_si = (si_it != pqi.mineral_si.end()) ? si_it->second : 0.0;
            min_comp.moles = (mol_it != pqi.mineral_moles.end()) ? mol_it->second : 0.0;
            cell.minerals.push_back(min_comp);
        }

        // Kinetics
        cell.kinetics = pqi.kinetics;
    }

    return input;
}

void SystemBuilder::equilibrate_initial_state(BackendSolver& solver, SystemInput& input) {
    int N = static_cast<int>(input.cells.size());
    const auto& matrices = solver.get_matrices();
    int E = matrices.E;
    int S = matrices.S;
    int P = matrices.P;

    // Stage 1: Solution speciation at fixed pH to compute charge imbalance
    solver.initialize_grid(input);
    bool conv1 = solver.solve_initial_equilibration(0, true, 0.0);
    if (!conv1) {
        std::cerr << "Warning: Stage 1 initial solution equilibration failed to converge.\n";
    }

    // Compute charge imbalance from Stage 1 converged species
    const auto& state1 = solver.get_state();
    std::vector<double> ln_act(E);
    for (int e = 0; e < E; ++e) ln_act[e] = state1.master_activities[e * N + 0];

    double I = 0.01;
    std::vector<double> molalities(S, 0.0);
    constexpr double LN10 = 2.302585092994045684;

    for (int step = 0; step < 4; ++step) {
        for (int i = 0; i < S; ++i) {
            double ln_m = LN10 * matrices.species_logK[i];
            for (int e = 0; e < E; ++e) {
                double coeff = matrices.species_stoich_matrix[i * E + e];
                if (coeff != 0.0) ln_m += coeff * ln_act[e];
            }
            double m = std::exp(ln_m);
            double gamma = debye_huckel_gamma_device(matrices.species_charge[i], I,
                                                     matrices.species_gamma_params[i * 2 + 0],
                                                     matrices.species_gamma_params[i * 2 + 1]);
            molalities[i] = m / gamma;
        }
        double sum_z2_m = 0.0;
        for (int i = 0; i < S; ++i) {
            sum_z2_m += (matrices.species_charge[i] * matrices.species_charge[i]) * molalities[i];
        }
        I = 0.5 * sum_z2_m;
        if (I < 1e-5) I = 1e-5;
    }

    double imbalance = 0.0;
    for (int i = 0; i < S; ++i) {
        imbalance += matrices.species_charge[i] * molalities[i];
    }

    // Stage 2: Add mineral masses into element totals and equilibrate with floating pH
    for (int idx = 0; idx < N; ++idx) {
        input.cells[idx].charge_balance_target = imbalance;

        for (int e = 0; e < E; ++e) {
            const std::string& e_name = matrices.elements[e];
            double tot = input.cells[idx].element_totals[e_name];
            for (int p = 0; p < P; ++p) {
                double coeff = matrices.mineral_stoich_matrix[p * E + e];
                if (coeff != 0.0) {
                    tot += coeff * input.cells[idx].minerals[p].moles;
                }
            }
            input.cells[idx].element_totals[e_name] = tot;
        }
    }

    // Re-initialize solver with updated totals and imbalance
    solver.initialize_grid(input);
    auto& state2 = solver.get_state();
    for (int idx = 0; idx < N; ++idx) {
        for (int e = 0; e < E; ++e) {
            state2.master_activities[e * N + idx] = ln_act[e];
        }
    }
    bool conv2 = solver.solve_initial_equilibration(0, false, imbalance);
    if (!conv2) {
        std::cerr << "Warning: Stage 2 initial mineral-solution equilibration failed to converge.\n";
    }

    // Re-speciate to get updated aqueous molalities and floating pH
    for (int e = 0; e < E; ++e) {
        ln_act[e] = state2.master_activities[e * N + 0];
    }
    I = 0.01;
    for (int step = 0; step < 4; ++step) {
        for (int i = 0; i < S; ++i) {
            double ln_m = LN10 * matrices.species_logK[i];
            for (int e = 0; e < E; ++e) {
                double coeff = matrices.species_stoich_matrix[i * E + e];
                if (coeff != 0.0) ln_m += coeff * ln_act[e];
            }
            double m = std::exp(ln_m);
            double gamma = debye_huckel_gamma_device(matrices.species_charge[i], I,
                                                     matrices.species_gamma_params[i * 2 + 0],
                                                     matrices.species_gamma_params[i * 2 + 1]);
            molalities[i] = m / gamma;
        }
        double sum_z2_m = 0.0;
        for (int i = 0; i < S; ++i) {
            sum_z2_m += (matrices.species_charge[i] * matrices.species_charge[i]) * molalities[i];
        }
        I = 0.5 * sum_z2_m;
        if (I < 1e-5) I = 1e-5;
    }

    double floating_ph = -ln_act[matrices.h_idx] / LN10;

    // Propagate converged state and true totals to all input cells
    for (int idx = 0; idx < N; ++idx) {
        input.cells[idx].initial_ph = floating_ph;
        for (int p = 0; p < P; ++p) {
            input.cells[idx].minerals[p].moles = state2.mineral_moles[p * N + 0];
        }
        for (int e = 0; e < E; ++e) {
            if (e == matrices.h_idx) continue;
            const std::string& e_name = matrices.elements[e];
            double tot = 0.0;
            for (int i = 0; i < S; ++i) {
                double coeff = matrices.species_stoich_matrix[i * E + e];
                if (coeff != 0.0) tot += coeff * molalities[i];
            }
            for (int p = 0; p < P; ++p) {
                double coeff = matrices.mineral_stoich_matrix[p * E + e];
                if (coeff != 0.0) tot += coeff * state2.mineral_moles[p * N + 0];
            }
            input.cells[idx].element_totals[e_name] = tot;
        }
    }

    // Final re-init with fully aligned initial state
    solver.initialize_grid(input);
    auto& final_state = solver.get_state();
    for (int idx = 0; idx < N; ++idx) {
        for (int e = 0; e < E; ++e) {
            final_state.master_activities[e * N + idx] = ln_act[e];
        }
        for (int p = 0; p < P; ++p) {
            final_state.mineral_moles[p * N + idx] = state2.mineral_moles[p * N + 0];
        }
    }
}

} // namespace geochem
