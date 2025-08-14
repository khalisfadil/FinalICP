#include <solver/gausnewtonsolvernva.hpp>

#include <iomanip>
#include <iostream>

#include <Eigen/Cholesky>

#include <common/timer.hpp>

namespace finalicp {

    // ###########################################################
    // GaussNewtonSolverNVA
    // ###########################################################

    GaussNewtonSolverNVA::GaussNewtonSolverNVA(Problem& problem, const Params& params)
    : GaussNewtonSolver(problem, params), params_(params) {}

    // ###########################################################
    // linearizeSolveAndUpdate
    // ###########################################################

    bool GaussNewtonSolverNVA::linearizeSolveAndUpdate(double& cost, double& grad_norm) {
        Timer iter_timer;
        Timer timer;
        double build_time = 0;
        double solve_time = 0;
        double update_time = 0;

        // The 'left-hand-side' of the Gauss-Newton problem, generally known as the
        // approximate Hessian matrix (note we only store the upper-triangular
        // elements)
        Eigen::SparseMatrix<double> approximate_hessian;
        // The 'right-hand-side' of the Gauss-Newton problem, generally known as the
        // gradient vector
        Eigen::VectorXd gradient_vector;

        // Construct system of equations
        timer.reset();
        problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
        grad_norm = gradient_vector.norm();
        build_time = timer.milliseconds();

#ifdef DEBUG
    // --- [KEY DEBUG] CHECK THE HEALTH OF THE LINEAR SYSTEM ---
    std::cout << "\n[000# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate]  ###################### START ####################. \n" << std::endl;
    std::cout << "[001# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] Built system. Grad norm: " << grad_norm << ". Hessian non-zeros: " << approximate_hessian.nonZeros() << std::endl;
    if (!gradient_vector.allFinite()) {
        std::cout << "[002# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] CRITICAL: Gradient vector contains non-finite values!" << std::endl;
        return false; // Abort this iteration
    }
    if (!approximate_hessian.coeffs().allFinite()) {
        std::cout << "[003# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] CRITICAL: Hessian matrix contains non-finite values!" << std::endl;
        return false; // Abort this iteration
    }
#endif

        // Solve system
        timer.reset();
        Eigen::VectorXd perturbation = solveGaussNewton(approximate_hessian, gradient_vector);
        solve_time = timer.milliseconds();

#ifdef DEBUG
    // --- [KEY DEBUG] CHECK THE HEALTH OF THE SOLUTION (THE STATE UPDATE) ---
    double perturbation_norm = perturbation.norm();
    std::cout << "[004# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] Solved system. Perturbation norm: " << perturbation_norm << std::endl;
    if (!perturbation.allFinite()) {
        std::cout << "[005# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] CRITICAL: Perturbation (state update) is non-finite! System solve failed." << std::endl;
        return false; // Abort this iteration
    }
#endif

        if (params_.line_search) {
            const double expected_delta_cost = 0.5 * gradient_vector.transpose() * perturbation;
            if (expected_delta_cost < 0.0) {
                throw std::runtime_error("Expected delta cost must be >= 0.0");
#ifdef DEBUG
                std::cout << "[006# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] CRITICAL: Expected delta cost is negative (" << expected_delta_cost << "). The descent direction is invalid." << std::endl;
#endif
            }
            if (expected_delta_cost < 1.0e-5 || fabs(expected_delta_cost / cost) < 1.0e-7) {
                solver_converged_ = true;
                term_ = TERMINATE_EXPECTED_DELTA_COST_CONVERGED;
            } else {
                double alpha = 1.0;
                for (uint j = 0; j < 3; ++j) {
                    timer.reset();
                    // Apply update
                    cost = proposeUpdate(alpha * perturbation);
                    update_time += timer.milliseconds();
#ifdef DEBUG
                    std::cout << "[007# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] Line search it: " << j << " prev_cost: " << prev_cost_ << " new_cost: " << cost << " alpha: " << alpha << std::endl;
#endif
                    if (cost <= prev_cost_) {
                        acceptProposedState();
                        break;
                    } else {
                        cost = prev_cost_;
                        rejectProposedState();
                    }
                    alpha *= 0.5;
                }
            }
        } else {
            // Apply update
            timer.reset();
            cost = proposeUpdate(perturbation);
            acceptProposedState();
            update_time = timer.milliseconds();
        }

#ifdef DEBUG
        // Print report line if verbose option enabled
        // clang-format off
        std::cout << "[008# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] Start Report: " << std::endl;
        std::cout << std::right << std::setw( 15) << std::setfill(' ') << "iter"
                    << std::right << std::setw(15) << std::setfill(' ') << "cost"
                    << std::right << std::setw(15) << std::setfill(' ') << "build (ms)"
                    << std::right << std::setw(15) << std::setfill(' ') << "solve (ms)"
                    << std::right << std::setw(15) << std::setfill(' ') << "update (ms)"
                    << std::right << std::setw(15) << std::setfill(' ') << "time (ms)"
                    << std::endl;
        // clang-format on
        
        // clang-format off
        std::cout << std::right << std::setw( 15) << std::setfill(' ') << curr_iteration_
                << std::right << std::setw(15) << std::setfill(' ') << std::setprecision(5) << cost
                << std::right << std::setw(15) << std::setfill(' ') << std::setprecision(3) << std::fixed << build_time << std::resetiosflags(std::ios::fixed)
                << std::right << std::setw(15) << std::setfill(' ') << std::setprecision(3) << std::fixed << solve_time << std::resetiosflags(std::ios::fixed)
                << std::right << std::setw(15) << std::setfill(' ') << std::setprecision(3) << std::fixed << update_time << std::resetiosflags(std::ios::fixed)
                << std::right << std::setw(15) << std::setfill(' ') << std::setprecision(3) << std::fixed << iter_timer.milliseconds() << std::resetiosflags(std::ios::fixed)
                << std::endl;
        std::cout << "[009# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate] End report." << std::endl;
        std::cout << "\n[000# GaussNewtonSolverNVA DEBUG | linearizeSolveAndUpdate]  ###################### END ####################. \n" << std::endl;
        // clang-format on
        
#endif

        return true;
    }

    // ###########################################################
    // solveGaussNewton
    // ###########################################################

    Eigen::VectorXd GaussNewtonSolverNVA::solveGaussNewton(const Eigen::SparseMatrix<double>& approximate_hessian, const Eigen::VectorXd& gradient_vector) {

#ifdef DEBUG
        // --- [PRE-SOLVE DIAGNOSTICS] ---
        // Check for obvious problems in the Hessian before attempting to solve.
        std::cout << "\n[000# GaussNewtonSolverNVA DEBUG | solveGaussNewton]  ###################### START ####################. \n" << std::endl;
        bool hessian_ok = true;
        for (int k = 0; k < approximate_hessian.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(approximate_hessian, k); it; ++it) {
                if (!std::isfinite(it.value())) {
                    std::cerr << "[001# GaussNewtonSolverNVA DEBUG | solveGaussNewton] CRITICAL: Hessian contains non-finite value at (" 
                            << it.row() << ", " << it.col() << ")!" << std::endl;
                    hessian_ok = false;
                    break;
                }
            }
            if (!hessian_ok) break;
        }

        for (int i = 0; i < approximate_hessian.rows(); ++i) {
            if (approximate_hessian.coeff(i, i) <= 0) {
                std::cerr << "[002# GaussNewtonSolverNVA DEBUG | solveGaussNewton] WARNING: Hessian has a non-positive diagonal at index "
                        << i << " (value: " << approximate_hessian.coeff(i, i) 
                        << "). This will cause Cholesky decomposition to fail." << std::endl;
                hessian_ok = false;
            }
        }

        if (!hessian_ok) {
            // Throw an exception to halt the optimization cleanly if pre-checks fail.
            throw decomp_failure("Pre-solve checks on the Hessian matrix failed.");
        }
#endif

        // --- [SYMBOLIC ANALYSIS] ---
        if (!pattern_initialized_) {
            hessian_solver_->analyzePattern(approximate_hessian);
            if (params_.reuse_previous_pattern) pattern_initialized_ = true;
        }

        // --- [NUMERICAL FACTORIZATION] ---
        hessian_solver_->factorize(approximate_hessian);

        // --- [FACTORIZATION SANITY CHECK] ---
        if (hessian_solver_->info() != Eigen::Success) {
            std::string error_msg = "[GaussNewtonSolverNVA DEBUG | solveGaussNewton] Eigen LLT decomposition failed. ";
            // Provide a more detailed error message in debug builds
            switch(hessian_solver_->info()) {
                case Eigen::NumericalIssue:
                    error_msg += "Reason: Numerical issue. The matrix is likely not positive-definite or is ill-conditioned.";
                    break;
                case Eigen::NoConvergence:
                    error_msg += "Reason: No convergence (not applicable to LLT, but for completeness).";
                    break;
                case Eigen::InvalidInput:
                    error_msg += "Reason: Invalid input.";
                    break;
                default:
                    error_msg += "Reason: Unknown.";
                    break;
            }

            throw decomp_failure(error_msg);
        }

        // --- [SOLVE & POST-SOLVE DIAGNOSTICS] ---
        Eigen::VectorXd perturbation = hessian_solver_->solve(gradient_vector);

#ifdef DEBUG
        // After solving, check the result. A non-finite or huge perturbation is a major red flag.
        if (!perturbation.allFinite()) {
            std::cerr << "[003# GaussNewtonSolverNVA DEBUG | solveGaussNewton] CRITICAL: The calculated perturbation vector contains non-finite values! The system is likely ill-conditioned." << std::endl;
        } else {
            std::cout << "[004# GaussNewtonSolverNVA DEBUG | solveGaussNewton] Calculated perturbation norm: " << perturbation.norm() << std::endl;
        }
        std::cout << "\n[000# GaussNewtonSolverNVA DEBUG | solveGaussNewton]  ###################### END ####################. \n" << std::endl;
#endif

        return perturbation;
    }
} // namespace finaleicp