#include <solver/gausnewtonsolver.hpp>

#include <iomanip>
#include <iostream>

#include <Eigen/Cholesky>

#include <common/timer.hpp>

namespace finalicp {

    GaussNewtonSolver::GaussNewtonSolver(Problem& problem, const Params& params)
    : SolverBase(problem, params), params_(params) {}

    bool GaussNewtonSolver::linearizeSolveAndUpdate(double& cost, double& grad_norm) {
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

        // Debug: Log Hessian and gradient sizes and norms
        // ################################
        std::cout << "[DEBUG] Iteration " << curr_iteration_ << ": Hessian size (" << approximate_hessian.rows() << "x" << approximate_hessian.cols() << "), non-zeros: " << approximate_hessian.nonZeros() << std::endl;
        std::cout << "[DEBUG] Iteration " << curr_iteration_ << ": Gradient size (" << gradient_vector.size() << "), norm: " << grad_norm << std::endl;
        // ################################

        // Solve system
        timer.reset();
        Eigen::VectorXd perturbation = solveGaussNewton(approximate_hessian, gradient_vector);
        solve_time = timer.milliseconds();

        // Debug: Log perturbation size and norm
        // ################################
        std::cout << "[DEBUG] Iteration " << curr_iteration_ << ": Perturbation size (" << perturbation.size() << "), norm: " << perturbation.norm() << std::endl;
        // ################################

        // Apply update
        timer.reset();
        cost = proposeUpdate(perturbation);
        acceptProposedState();
        update_time = timer.milliseconds();

        // Debug: Log updated cost
        // ################################
        std::cout << "[DEBUG] Iteration " << curr_iteration_ << ": Updated cost: " << cost << std::endl;
        // ################################

        // Print report line if verbose option enabled
        // if (params_.verbose) {
        //     if (curr_iteration_ == 1) {
        //     // clang-format off
        //     std::cout << std::right << std::setw( 4) << std::setfill(' ') << "iter"
        //                 << std::right << std::setw(12) << std::setfill(' ') << "cost"
        //                 << std::right << std::setw(12) << std::setfill(' ') << "build (ms)"
        //                 << std::right << std::setw(12) << std::setfill(' ') << "solve (ms)"
        //                 << std::right << std::setw(13) << std::setfill(' ') << "update (ms)"
        //                 << std::right << std::setw(11) << std::setfill(' ') << "time (ms)"
        //                 << std::endl;
        //     // clang-format on
        //     }
        //     // clang-format off
        //     std::cout << std::right << std::setw( 4) << std::setfill(' ') << curr_iteration_
        //             << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(5) << cost
        //             << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(3) << std::fixed << build_time << std::resetiosflags(std::ios::fixed)
        //             << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(3) << std::fixed << solve_time << std::resetiosflags(std::ios::fixed)
        //             << std::right << std::setw(13) << std::setfill(' ') << std::setprecision(3) << std::fixed << update_time << std::resetiosflags(std::ios::fixed)
        //             << std::right << std::setw(11) << std::setfill(' ') << std::setprecision(3) << std::fixed << iter_timer.milliseconds() << std::resetiosflags(std::ios::fixed)
        //             << std::endl;
        //     // clang-format on
        // }

        return true;
    }

    Eigen::VectorXd GaussNewtonSolver::solveGaussNewton(const Eigen::SparseMatrix<double>& approximate_hessian, const Eigen::VectorXd& gradient_vector) {
        // Perform a Cholesky factorization of the approximate Hessian matrix
        // Check if the pattern has been initialized
        if (!pattern_initialized_) {
            hessian_solver_->analyzePattern(approximate_hessian);
            if (params_.reuse_previous_pattern) pattern_initialized_ = true;
        }

        // Perform a Cholesky factorization of the approximate Hessian matrix
        hessian_solver_->factorize(approximate_hessian);

        // Check if the factorization succeeded
        if (hessian_solver_->info() != Eigen::Success) {
            throw decomp_failure(
                "During steam solve, Eigen LLT decomposition failed. "
                "It is possible that the matrix was ill-conditioned, in which case "
                "adding a prior may help. On the other hand, it is also possible that "
                "the problem you've constructed is not positive semi-definite.");
        }

        // Do the backward pass, using the Cholesky factorization (fast)
        return hessian_solver_->solve(gradient_vector);
    }
} // namespace finaleicp