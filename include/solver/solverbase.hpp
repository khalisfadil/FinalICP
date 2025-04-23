#pragma once

#include <stdexcept>

#include <Eigen/Core>

#include <problem/problem.hpp>

namespace finalicp{
    //Exception for solver failures (e.g., numerical issues like LLT decomposition failure).
    class solver_failure : public std::runtime_error {
        public:
            solver_failure(const std::string& s) : std::runtime_error(s) {}
    }; 
    
    //Exception for unsuccessful optimization steps.
    class unsuccessful_step : public solver_failure {
        public:
            unsuccessful_step(const std::string& s) : solver_failure(s) {}
    };  
    
    //Abstract base class for nonlinear optimization solvers.
    class SolverBase {
        public:
            struct Params {
                virtual ~Params() = default;
                bool verbose = false;                               //Enable verbose logging for debugging solver iterations.
                unsigned int max_iterations = 100;                  //Maximum number of solver iterations before termination.
                double absolute_cost_threshold = 0.0;               //Absolute cost threshold for convergence.
                double absolute_cost_change_threshold = 1e-4;       //Convergence criterion based on the absolute change in cost.
                double relative_cost_change_threshold = 1e-4;       //Convergence criterion based on the relative change in cost.
            };

            enum Termination {
                TERMINATE_NOT_YET_TERMINATED,                       //Solver is still running.
                TERMINATE_STEP_UNSUCCESSFUL,                        //Step failure (e.g., numerical instability).
                TERMINATE_MAX_ITERATIONS,                           //Maximum iteration count reached.
                TERMINATE_CONVERGED_ABSOLUTE_ERROR,                 //Cost function met absolute error threshold.
                TERMINATE_CONVERGED_ABSOLUTE_CHANGE,                //Cost function change below absolute threshold.
                TERMINATE_CONVERGED_RELATIVE_CHANGE,                //Cost function change below relative threshold.
                TERMINATE_CONVERGED_ZERO_GRADIENT,                  //Optimization stopped due to zero gradient.
                TERMINATE_COST_INCREASED,                           //Cost unexpectedly increased (possible divergence).
                TERMINATE_EXPECTED_DELTA_COST_CONVERGED,            //Expected cost reduction is too small.
            };

            //Constructs the solver with a given problem and solver parameters.
            SolverBase(Problem& problem, const Params& params);

            //Default virtual destructor
            virtual ~SolverBase() = default;

            //Returns the reason why the solver terminated
            Termination termination_cause() const { return term_; }
            
            //Returns the current iteration number
            unsigned int curr_iteration() const { return curr_iteration_; }

            //uns the solver optimization process.
            void optimize();

        protected:

            //Performs a single iteration of the optimization process.
            void iterate();

            //Computes and proposes an update to the state vector.
            double proposeUpdate(const Eigen::VectorXd& perturbation);

            //Accepts the proposed state update if it improves the solution.
            void acceptProposedState();

            //Rejects the proposed state update, reverting to the previous state.
            void rejectProposedState();

            //Returns a constant reference to the state vector.
            StateVector::ConstWeakPtr state_vector() { return state_vector_; }

            //Reference to the optimization problem
            Problem& problem_;

            //Pointer to the current state vector
            const StateVector::WeakPtr state_vector_;

            //Backup state vector for reverting to previous values
            StateVector state_vector_backup_;

            //Solver state variables
            Termination term_ = TERMINATE_NOT_YET_TERMINATED;

            unsigned int curr_iteration_ = 0;

            bool solver_converged_ = false;

            double curr_cost_ = 0.0;

            double prev_cost_ = 0.0;

            bool pending_proposed_state_ = false;

        private:

            //Virtual method to perform linearization, solve, and update.
            virtual bool linearizeSolveAndUpdate(double& cost, double& grad_norm) = 0;

            //Solver parameters
            const Params params_;
    };
    
    //Overloads the output stream operator for `Termination` enum.
    std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T);

} // namespace finaleicp