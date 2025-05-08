#include <solver/solverbase.hpp>
#include <iostream>
#include <common/timer.hpp>

namespace finalicp {

    SolverBase::SolverBase(Problem& problem, const Params& params)
        : problem_(problem),
          state_vector_(problem.getStateVector()),
          params_(params) {
        state_vector_backup_ = state_vector_.lock()->clone();
        curr_cost_ = prev_cost_ = problem_.cost();

        //debug
        std::cout << "curr_cost_: "  << curr_cost_ << std::endl;
    }

    void SolverBase::optimize() {
        Timer timer;
        while (!solver_converged_) iterate();
        if (params_.verbose)
            std::cout << "Total Optimization Time: " << timer.milliseconds() << " ms" << std::endl;
    }

    void SolverBase::iterate() {
        if (solver_converged_) {
            std::cout << "Requested an iteration when solver has already converged, iteration ignored.";
            return;
        }

        if (params_.verbose && curr_iteration_ == 0) {
            std::cout << "Begin Optimization" << std::endl;
            std::cout << "------------------" << std::endl;
            std::cout << "Number of States: " << state_vector_.lock()->getNumberOfStates() << std::endl;
            std::cout << "Number of Cost Terms: " << problem_.getNumberOfCostTerms() << std::endl;
            std::cout << "Initial Cost: " << curr_cost_ << std::endl;
        }

        curr_iteration_++;
        prev_cost_ = curr_cost_;
        double grad_norm = 0.0;

        //debug
        std::cout << "start linearizeSolveAndUpdate ..."  << std::endl;
        
        bool step_success = linearizeSolveAndUpdate(curr_cost_, grad_norm);

        if (!step_success && fabs(grad_norm) < 1e-6) {
            term_ = TERMINATE_CONVERGED_ZERO_GRADIENT;
            solver_converged_ = true;
        } else if (!step_success) {
            term_ = TERMINATE_STEP_UNSUCCESSFUL;
            solver_converged_ = true;
            throw unsuccessful_step(
                "The steam solver terminated due to being unable to produce a "
                "'successful' step. If this occurs, it is likely that your problem "
                "is very nonlinear and poorly initialized, or is using incorrect "
                "analytical Jacobians.");
        } else if (curr_iteration_ >= params_.max_iterations) {
            term_ = TERMINATE_MAX_ITERATIONS;
            solver_converged_ = true;
        } else if (curr_cost_ <= params_.absolute_cost_threshold) {
            term_ = TERMINATE_CONVERGED_ABSOLUTE_ERROR;
            solver_converged_ = true;
        } else if (fabs(prev_cost_ - curr_cost_) <= params_.absolute_cost_change_threshold) {
            term_ = TERMINATE_CONVERGED_ABSOLUTE_CHANGE;
            solver_converged_ = true;
        } else if (fabs(prev_cost_ - curr_cost_) / prev_cost_ <= params_.relative_cost_change_threshold) {
            term_ = TERMINATE_CONVERGED_RELATIVE_CHANGE;
            solver_converged_ = true;
        }

        if (params_.verbose && solver_converged_)
            std::cout << "Termination Cause: " << term_ << std::endl;
    }

    double SolverBase::proposeUpdate(const Eigen::VectorXd& perturbation) {
        if (pending_proposed_state_) {
            throw std::runtime_error(
                "There is already a pending update, accept "
                "or reject before proposing a new one.");
        }
        const auto state_vector = state_vector_.lock();
        if (!state_vector) throw std::runtime_error{"state vector expired"};
        state_vector_backup_.copyValues(*(state_vector));
        state_vector->update(perturbation);
        pending_proposed_state_ = true;
        return problem_.cost();
    }

    void SolverBase::acceptProposedState() {
        if (!pending_proposed_state_)
            throw std::runtime_error("You must call proposeUpdate before accept.");
        pending_proposed_state_ = false;
    }

    void SolverBase::rejectProposedState() {
        if (!pending_proposed_state_)
            throw std::runtime_error("You must call proposeUpdate before rejecting.");
        state_vector_.lock()->copyValues(state_vector_backup_);
        pending_proposed_state_ = false;
    }

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