#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <sycl/sycl.hpp>
#include "dto.hpp"
#include "parser.hpp"
#include "solver_backend.hpp"

bool check_close(const std::string& name, double actual, double expected, double rel_tol = 0.01, double abs_tol = 1e-6) {
    double diff = std::fabs(actual - expected);
    double tol = std::max(abs_tol, rel_tol * std::fabs(expected));
    bool ok = diff <= tol;
    std::cout << "  " << std::left << std::setw(18) << name << " | Expected: "
              << std::setw(12) << expected << " | Actual: " << std::setw(12) << actual
              << " | " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  Running Automated Verification Test (test_verify)\n";
    std::cout << "=========================================================\n";

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

    std::string db_path = "database/phreeqc_kin.dat";
    std::string pqi_path = "python/verify.pqi";
    std::ifstream test_f(db_path);
    if (!test_f.good()) {
        db_path = "../../database/phreeqc_kin.dat";
        pqi_path = "../../python/verify.pqi";
    }

    geochem::PhreeqcDatabase db;
    if (!db.parse(db_path)) {
        std::cerr << "FAILED: Unable to parse " << db_path << "\n";
        return 1;
    }

    geochem::PqiParser pqi;
    if (!pqi.parse_pqi(pqi_path, db)) {
        std::cerr << "FAILED: Unable to parse " << pqi_path << "\n";
        return 1;
    }

    constexpr int N = 10; // Test with 10 cells in parallel
    auto matrices = geochem::SystemBuilder::build_system_matrices(db, pqi);
    auto input = geochem::SystemBuilder::create_system_input(db, pqi, matrices, N);

    geochem::BackendSolver solver(q, matrices);
    geochem::SystemBuilder::equilibrate_initial_state(solver, input);

    auto output = solver.run_simulation(input.dt);

    std::cout << "Verification Benchmark vs PHREEQC Reference (python/verify.out):\n";
    std::cout << "-----------------------------------------------------------------\n";

    bool all_ok = true;
    const auto& res0 = output.cells[0];

    // Reference values from verify.out:
    // Final pH = 7.005
    // Final Ca = 1.0028e-3
    // Final C  = 2.003e-3 (or 2.0028e-3)
    // Final S  = 3.000e-3
    // Final Calcite = 9.9972e-3
    // Final Gypsum = 0.0

    all_ok &= check_close("Final pH", res0.final_ph, 7.00503, 0.005);
    all_ok &= check_close("Final Ca (mol/kgw)", res0.element_totals.at("Ca+2"), 1.0028e-3, 0.01);
    all_ok &= check_close("Final C (mol/kgw)", res0.element_totals.at("CO3-2"), 2.0028e-3, 0.01);
    all_ok &= check_close("Final S (mol/kgw)", res0.element_totals.at("SO4-2"), 3.0000e-3, 0.005);
    all_ok &= check_close("Final Calcite (mol)", res0.kinetic_moles.at("Calcite"), 9.9972e-3, 0.005);
    all_ok &= check_close("Final Gypsum (mol)", res0.mineral_moles.at("Gypsum"), 0.0, 0.01, 1e-6);

    // Multi-cell consistency check
    std::cout << "\nChecking multi-cell consistency across " << N << " cells...\n";
    bool multi_ok = true;
    for (int i = 1; i < N; ++i) {
        if (std::fabs(output.cells[i].final_ph - res0.final_ph) > 1e-5 ||
            std::fabs(output.cells[i].element_totals.at("Ca+2") - res0.element_totals.at("Ca+2")) > 1e-8) {
            multi_ok = false;
            break;
        }
    }
    std::cout << "Multi-cell consistency: " << (multi_ok ? "PASS" : "FAIL") << "\n";
    all_ok &= multi_ok;

    std::cout << "=========================================================\n";
    if (all_ok) {
        std::cout << ">>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<\n";
        return 0;
    } else {
        std::cerr << ">>> VERIFICATION TESTS FAILED! <<<\n";
        return 1;
    }
}
