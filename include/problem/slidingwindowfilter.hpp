#pragma once

#include <deque>
#include <problem/problem.hpp>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>

namespace finalicp {
    class SlidingWindowFilter : public Problem {

        struct Variable {
            Variable(const StateVarBase::Ptr& v, bool m)
                : variable(v), marginalize(m) {}
            StateVarBase::Ptr variable = nullptr;
            bool marginalize = false;
        };

        //Maps state keys to their corresponding variables.
        using VariableMap = std::unordered_map<StateKey, Variable, StateKeyHash>;

        using KeySet = BaseCostTerm::KeySet;  //KeySet for tracking variable dependencies

        public:

            using Ptr = std::shared_ptr<SlidingWindowFilter>;

            static Ptr MakeShared(unsigned int num_threads = 1);

            //Constructor for Sliding Window Filter.
            SlidingWindowFilter(unsigned int num_threads = 1);

            //Debugging
            const VariableMap& variables() const { return variables_; }

            //Adds a state variable to the sliding window.
            void addStateVariable(const StateVarBase::Ptr& variable) override;

            //Adds a state variable to the sliding window.
            void addStateVariable(const std::vector<StateVarBase::Ptr>& variables);

            //Marks a state variable for marginalization.
            void marginalizeVariable(const StateVarBase::Ptr& variable);

            //Marks a state variable for marginalization.
            void marginalizeVariable(const std::vector<StateVarBase::Ptr>& variables);

            //Adds a cost term to the sliding window filter.
            void addCostTerm(const BaseCostTerm::ConstPtr& cost_term) override;
            
            //Computes the total cost of the optimization problem.
            double cost() const override;

            //Retrieves the total number of cost terms.
            unsigned int getNumberOfCostTerms() const override;

            //Retrieves the total number of cost terms.
            unsigned int getNumberOfVariables() const;

            //Retrieves the active state vector (excluding marginalized variables).
            StateVector::Ptr getStateVector() const override;

            //Constructs the Gauss-Newton system.
            void buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian,
                             Eigen::VectorXd& gradient_vector) const override;

        private:
            //Cumber of threads to evaluate cost terms
            const unsigned int num_threads_;

            //Represents a state variable in the filter.
            VariableMap variables_;

            //Maintains an ordered queue of state variables for the sliding window. */
            std::deque<StateKey> variable_queue_;

            //Tracks variable dependencies for marginalization.
            std::unordered_map<StateKey, KeySet, StateKeyHash> related_var_keys_;

            //Collection of cost terms affecting the current optimization window.
            std::vector<BaseCostTerm::ConstPtr> cost_terms_;

            //Active state vector (only contains non-marginalized variables)
            const StateVector::Ptr marginalize_state_vector_ = StateVector::MakeShared();

            //State vector containing marginalized variables.
            const StateVector::Ptr active_state_vector_ = StateVector::MakeShared();

            //Complete state vector containing both active and marginalized variables.
            const StateVector::Ptr state_vector_ = StateVector::MakeShared();

            //Fixed linearized system from marginalized variables (stored as dense for now). */
            Eigen::MatrixXd fixed_A_;

            //Fixed linearized system from marginalized variables (stored as dense for now). */
            Eigen::VectorXd fixed_b_;
    };
}   // namespace finalicp