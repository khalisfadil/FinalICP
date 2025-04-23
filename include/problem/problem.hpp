#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <problem/costterm/basecostterm.hpp>
#include <problem/statevector.hpp>

namespace finalicp {
    //Interface for a SLAM optimization problem.
    class Problem {
        public:
            virtual ~Problem() = default;
            using Ptr = std::shared_ptr<Problem>;

            //Retrieves the total number of cost terms.
            virtual unsigned int getNumberOfCostTerms() const = 0;

            //Adds a state variable to the problem.
            virtual void addStateVariable(const StateVarBase::Ptr& state_var) = 0;

            //Adds a cost term to the optimization problem.
            virtual void addCostTerm(const BaseCostTerm::ConstPtr& cost_term) = 0;

            //Computes the total cost of the optimization problem.
            virtual double cost() const = 0;

            //Retrieves a reference to the state vector.
            virtual StateVector::Ptr getStateVector() const = 0;

            //Constructs the Gauss-Newton system.
            virtual void buildGaussNewtonTerms(
                Eigen::SparseMatrix<double>& approximate_hessian,
                Eigen::VectorXd& gradient_vector) const = 0;
    };
} // namespace finalicp