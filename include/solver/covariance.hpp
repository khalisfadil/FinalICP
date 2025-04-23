#pragma once

#include <problem/problem.hpp>
#include <solver/gausnewtonsolver.hpp>

namespace finalicp {

    //Implements the Gauss-Newton optimization algorithm.
    class Covariance {

        public:
            using Ptr = std::shared_ptr<Covariance>;
            using ConstPtr = std::shared_ptr<const Covariance>;

            //Constructs covariance estimation from a problem
            Covariance(Problem& problem);

            //Constructs covariance estimation from a solver
            Covariance(GaussNewtonSolver& solver);

            virtual ~Covariance() = default;

            //Queries covariance of a single state variable
            Eigen::MatrixXd query(const StateVarBase::ConstPtr& var) const;

            //Queries covariance between two state variables
            Eigen::MatrixXd query(const StateVarBase::ConstPtr& rvar, const StateVarBase::ConstPtr& cvar) const;

            //Queries joint covariance of multiple variables
            Eigen::MatrixXd query(const std::vector<StateVarBase::ConstPtr>& vars) const;
            
            //Queries block covariance between row and column variables
            Eigen::MatrixXd query(const std::vector<StateVarBase::ConstPtr>& rvars, const std::vector<StateVarBase::ConstPtr>& cvars) const;

         private:

            const StateVector::ConstWeakPtr state_vector_;

            using SolverType = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>;

            const std::shared_ptr<SolverType> hessian_solver_;
    };
} // namespace finaleicp