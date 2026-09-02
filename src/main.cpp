#include <iostream>
#include <iomanip>
#include <string>
#include <sycl/sycl.hpp>
#include "dto.hpp"
#include "parser.hpp"
#include "solver_backend.hpp"

int main(int argc, char* argv[]) {
    std::string db_path = "database/phreeqc_kin.dat";
    std::string pqi_path = "python/verify.pqi";
    int num_cells = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-db" || arg == "--database") && i + 1 < argc) {
            db_path = argv[++i];
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            pqi_path = argv[++i];
        } else if ((arg == "-n" || arg == "--num_cells") && i + 1 < argc) {
            num_cells = std::stoi(argv[++i]);
        }
    }

    std::cout << "=========================================================\n";
    std::cout << "  C++ SYCL Parallel Geochemical Solver (AdaptiveCpp)\n";
    std::cout << "=========================================================\n";

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::cout << "Target Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Database:      " << db_path << "\n";
    std::cout << "Input Script:  " << pqi_path << "\n";
    std::cout << "Cells Count:   " << num_cells << "\n\n";

    // 1. Parse database and PQI
    geochem::PhreeqcDatabase db;
    if (!db.parse(db_path)) return 1;

    geochem::PqiParser pqi;
    if (!pqi.parse_pqi(pqi_path, db)) return 1;

    // 2. Build system matrices and DTOs
    auto matrices = geochem::SystemBuilder::build_system_matrices(db, pqi);
    auto input = geochem::SystemBuilder::create_system_input(db, pqi, matrices, num_cells);

    std::cout << "Chemical System:\n";
    std::cout << "  Elements: ";
    for (const auto& e : matrices.elements) std::cout << e << " ";
    std::cout << "\n  Equilibrium Minerals: ";
    for (const auto& m : matrices.mineral_names) std::cout << m << " ";
    std::cout << "\n  Kinetics: ";
    for (const auto& k : matrices.kinetics_names) std::cout << k << " ";
    std::cout << "\n\n";

    // 3. Initialize solver & execute two-stage equilibration
    geochem::BackendSolver solver(q, matrices);
    std::cout << "Performing two-stage initial equilibration...\n";
    geochem::SystemBuilder::equilibrate_initial_state(solver, input);

    double initial_ph = input.cells[0].initial_ph;
    std::cout << "Equilibrated Initial pH: " << std::fixed << std::setprecision(5) << initial_ph << "\n\n";

    // 4. Run parallel simulation
    std::cout << "Launching SYCL parallel kernel (dt = " << input.dt << " s across " << num_cells << " cells)...\n";
    auto output = solver.run_simulation(input.dt);

    std::cout << "Simulation completed in " << std::setprecision(4) << output.elapsed_seconds << " seconds.\n";
    if (output.elapsed_seconds > 0.0) {
        std::cout << "Throughput: " << std::setprecision(1) << (num_cells / output.elapsed_seconds) << " cells/sec\n";
    }

    // 5. Report Cell 0 Results
    std::cout << "\n--- Final Simulation State (Cell 0) ---\n";
    const auto& res0 = output.cells[0];
    std::cout << "Status: " << (res0.converged ? "CONVERGED" : "FAILED") << "\n";
    std::cout << "Final pH: " << std::fixed << std::setprecision(5) << res0.final_ph << "\n\n";

    std::cout << "Elements (mol/kgw):\n";
    for (const auto& e : matrices.elements) {
        if (e != "H+") {
            std::cout << "  " << std::left << std::setw(8) << e << " : "
                      << std::scientific << std::setprecision(6) << res0.element_totals.at(e) << "\n";
        }
    }

    std::cout << "\nEquilibrium Minerals (mol):\n";
    for (const auto& m : matrices.mineral_names) {
        std::cout << "  " << std::left << std::setw(8) << m << " : "
                  << std::scientific << std::setprecision(6) << res0.mineral_moles.at(m) << "\n";
    }

    std::cout << "\nKinetics (mol):\n";
    for (const auto& k : matrices.kinetics_names) {
        std::cout << "  " << std::left << std::setw(8) << k << " : "
                  << std::scientific << std::setprecision(6) << res0.kinetic_moles.at(k) << "\n";
    }
    std::cout << "=========================================================\n";

    return output.all_converged ? 0 : 1;
}
