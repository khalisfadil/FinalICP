#pragma once

#include <problem/problem.hpp>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

namespace finalicp {
    class OptimizationProblem : public Problem {
        public:
            using Ptr = std::shared_ptr<OptimizationProblem>;
            static Ptr MakeShared(unsigned int num_threads = 1);
            OptimizationProblem(unsigned int num_threads = 1);

            //Adds a state variable
            void addStateVariable(const StateVarBase::Ptr& state_var) override;

            //Add a cost term
            void addCostTerm(const BaseCostTerm::ConstPtr& cost_term) override;

            //Get the total number of cost terms
            unsigned int getNumberOfCostTerms() const override;

            //Compute the cost from the collection of cost terms
            double cost() const override;

            //Get reference to state variables
            StateVector::Ptr getStateVector() const override;

            //Fill in the supplied block matrices
            void buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian,
                             Eigen::VectorXd& gradient_vector) const override;

        private:

            //Cumber of threads to evaluate cost terms
            const unsigned int num_threads_;

            //Collection of cost terms
            std::vector<BaseCostTerm::ConstPtr> cost_terms_;

            //Collection of state variables
            std::vector<StateVarBase::Ptr> state_vars_;

            //State vector, created when calling get state vector
            StateVector::Ptr state_vector_ = StateVector::MakeShared();
    };
} // namespace finalicp