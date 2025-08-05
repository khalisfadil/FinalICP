#include <solver/solverbase.hpp>
#include <iostream>
#include <common/timer.hpp>

namespace finalicp {

    // ###############################################################################
    // SolverBase
    // ###############################################################################

    SolverBase::SolverBase(Problem& problem, const Params& params)
        : problem_(problem),
          state_vector_(problem.getStateVector()),
          params_(params) {
        state_vector_backup_ = state_vector_.lock()->clone();
        curr_cost_ = prev_cost_ = problem_.cost();

#ifdef DEBUG
        // --- [IMPROVEMENT] VALIDATE INITIAL COST ---
        // It's crucial to know if the problem is valid from the very start.
        if (!std::isfinite(curr_cost_)) {
            std::cerr << "[SOLVERBASE DEBUG] CRITICAL: Initial cost is not finite (inf or NaN)!" << std::endl;
            throw std::runtime_error("Solver initialized with a non-finite cost.");
        }
#endif
    }

    // ###############################################################################
    // optimize
    // ###############################################################################

    void SolverBase::optimize() {
        Timer timer;
        while (!solver_converged_) iterate();

#ifdef DEBUG
            std::cout << "[SOLVERBASE DEBUG | optimize] Total Optimization Time: " << timer.milliseconds() << " ms" << std::endl;
#endif
    }

    // ###############################################################################
    // iterate
    // ###############################################################################

    void SolverBase::iterate() {
        if (solver_converged_) {
#ifdef DEBUG
            std::cout << "[SOLVERBASE DEBUG | iterate] Terminating: Requested an iteration when solver has already converged, iteration ignored.";
#endif
            return;
        }

#ifdef DEBUG
        if (curr_iteration_ == 0) {
            std::cout << "[SOLVERBASE DEBUG | iterate] Begin Optimization" << std::endl;
            std::cout << "[SOLVERBASE DEBUG | iterate] Number of States: " << state_vector_.lock()->getNumberOfStates() << std::endl;
            std::cout << "[SOLVERBASE DEBUG | iterate] Number of Cost Terms: " << problem_.getNumberOfCostTerms() << std::endl;
            std::cout << "[SOLVERBASE DEBUG | iterate] Initial Cost: " << curr_cost_ << std::endl;
        }
#endif

        curr_iteration_++;
        prev_cost_ = curr_cost_;
        double grad_norm = 0.0;
        bool step_success = linearizeSolveAndUpdate(curr_cost_, grad_norm);

         // --- [IMPROVEMENT] ADDED MORE TRANSPARENT TERMINATION LOGIC ---
        // This section now clearly states why the solver is stopping.
        const double cost_change = fabs(prev_cost_ - curr_cost_);
        const double rel_cost_change = (prev_cost_ > 1.0e-9) ? (cost_change / prev_cost_) : 0.0;

        if (!step_success && fabs(grad_norm) < 1e-6) {
            term_ = TERMINATE_CONVERGED_ZERO_GRADIENT;
            solver_converged_ = true;
#ifdef DEBUG
            if (params_.verbose) std::cout << "[SOLVERBASE DEBUG | iterate] Terminating: Step failed but gradient norm is near zero (" << grad_norm << ")." << std::endl;
#endif
        } else if (!step_success) {
            term_ = TERMINATE_STEP_UNSUCCESSFUL;
            solver_converged_ = true;
            throw unsuccessful_step(
                "The solver terminated due to being unable to produce a 'successful' step. "
                "This likely means the problem is very nonlinear, poorly initialized, or has incorrect analytical Jacobians.");
        } else if (curr_iteration_ >= params_.max_iterations) {
            term_ = TERMINATE_MAX_ITERATIONS;
            solver_converged_ = true;
#ifdef DEBUG
            if (params_.verbose) std::cout << "[SOLVERBASE DEBUG] Terminating: Max iterations (" << curr_iteration_ << ") reached." << std::endl;
#endif
        } else if (curr_cost_ > prev_cost_ && cost_change > params_.absolute_cost_change_threshold) {
            // --- [IMPROVEMENT] ADDED DIVERGENCE CHECK ---
            // This is a critical check that was missing. If cost increases, we should stop.
            term_ = TERMINATE_COST_INCREASED;
            solver_converged_ = true;
#ifdef DEBUG
            if (params_.verbose) std::cout << "[SOLVERBASE DEBUG| iterate] Terminating: Cost increased from " << prev_cost_ << " to " << curr_cost_ << ". (Divergence)" << std::endl;
#endif
        } else if (curr_cost_ <= params_.absolute_cost_threshold) {
            term_ = TERMINATE_CONVERGED_ABSOLUTE_ERROR;
            solver_converged_ = true;
#ifdef DEBUG
            if (params_.verbose) std::cout << "[SOLVERBASE DEBUG | iterate] Terminating: Cost (" << curr_cost_ << ") is below absolute threshold (" << params_.absolute_cost_threshold << ")." << std::endl;
#endif
        } else if (cost_change <= params_.absolute_cost_change_threshold) {
            term_ = TERMINATE_CONVERGED_ABSOLUTE_CHANGE;
            solver_converged_ = true;
#ifdef DEBUG
            if (params_.verbose) std::cout << "[SOLVERBASE DEBUG| iterate] Terminating: Cost change (" << cost_change << ") is below absolute threshold (" << params_.absolute_cost_change_threshold << ")." << std::endl;
#endif
        } else if (rel_cost_change <= params_.relative_cost_change_threshold) {
            term_ = TERMINATE_CONVERGED_RELATIVE_CHANGE;
            solver_converged_ = true;
#ifdef DEBUG
            if (params_.verbose) std::cout << "[SOLVERBASE DEBUG| iterate] Terminating: Relative cost change (" << rel_cost_change << ") is below relative threshold (" << params_.relative_cost_change_threshold << ")." << std::endl;
#endif
        }

        if (params_.verbose && solver_converged_)
            std::cout << "Termination Cause: " << term_ << std::endl;
    }

    // ###############################################################################
    // proposeUpdate
    // ###############################################################################

    double SolverBase::proposeUpdate(const Eigen::VectorXd& perturbation) {
        if (pending_proposed_state_) {
            throw std::runtime_error("There is already a pending update, accept or reject before proposing a new one.");
        }
        const auto state_vector = state_vector_.lock();
        if (!state_vector) throw std::runtime_error{"state vector expired"};
        state_vector_backup_.copyValues(*(state_vector));
        state_vector->update(perturbation);
        pending_proposed_state_ = true;
        
        const double new_cost = problem_.cost();
#ifdef DEBUG
        // --- [IMPROVEMENT] INVALID UPDATE DETECTION ---
        // Checks if the proposed state update resulted in an invalid cost.
        if (!std::isfinite(new_cost)) {
             std::cerr << "[SOLVERBASE DEBUG | proposeUpdate] CRITICAL: Proposed update resulted in a non-finite cost! The perturbation may be invalid or too large." << std::endl;
        }
#endif
        return new_cost;
    }

    // ###############################################################################
    // acceptProposedState
    // ###############################################################################

    void SolverBase::acceptProposedState() {
        if (!pending_proposed_state_)
            throw std::runtime_error("You must call proposeUpdate before accept.");
        pending_proposed_state_ = false;
    }

    // ###############################################################################
    // rejectProposedState
    // ###############################################################################

    void SolverBase::rejectProposedState() {
        if (!pending_proposed_state_)
            throw std::runtime_error("You must call proposeUpdate before rejecting.");
        state_vector_.lock()->copyValues(state_vector_backup_);
        pending_proposed_state_ = false;
    }

    // ###############################################################################
    // operator
    // ###############################################################################

    std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T) {
        switch (T) {
            case SolverBase::TERMINATE_NOT_YET_TERMINATED:
                out << "NOT YET TERMINATED";
                break;
            case SolverBase::TERMINATE_STEP_UNSUCCESSFUL:
                out << "STEP UNSUCCESSFUL";
                break;
            case SolverBase::TERMINATE_MAX_ITERATIONS:
                out << "MAX ITERATIONS";
                break;
            case SolverBase::TERMINATE_CONVERGED_ABSOLUTE_ERROR:
                out << "CONVERGED ABSOLUTE ERROR";
                break;
            case SolverBase::TERMINATE_CONVERGED_ABSOLUTE_CHANGE:
                out << "CONVERGED ABSOLUTE CHANGE";
                break;
            case SolverBase::TERMINATE_CONVERGED_RELATIVE_CHANGE:
                out << "CONVERGED RELATIVE CHANGE";
                break;
            case SolverBase::TERMINATE_CONVERGED_ZERO_GRADIENT:
                out << "CONVERGED GRADIENT IS ZERO";
                break;
            case SolverBase::TERMINATE_COST_INCREASED:
                out << "COST INCREASED";
                break;
            case SolverBase::TERMINATE_EXPECTED_DELTA_COST_CONVERGED:
                out << "CONVERGED EXPECTED DELTA COST";
                break;
        }
        return out;
    }

} // namespace finalicp