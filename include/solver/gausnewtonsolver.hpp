#pragma once

#include <Eigen/Core>

#include <matrixoperator/matrix.hpp>
#include <matrixoperator/matrixsparse.hpp>
#include <matrixoperator/vector.hpp>

#include <solver/solverbase.hpp>

namespace finalicp {

    //Exception class for decomposition failures.
    class decomp_failure : public solver_failure {
        public:
            decomp_failure(const std::string& s) : solver_failure(s) {}
    };

    //Implements the Gauss-Newton optimization algorithm.
    class GaussNewtonSolver : public SolverBase {

        public:
            //Configuration parameters for the Gauss-Newton solver.
            struct Params : public SolverBase::Params {bool reuse_previous_pattern = true;};

            //Constructs the Gauss-Newton solver.
            GaussNewtonSolver(Problem& problem, const Params& params);

        protected:

            //Solves the Gauss-Newton system: `Hessian * x = gradient`
            Eigen::VectorXd solveGaussNewton(const Eigen::SparseMatrix<double>& approximate_hessian,const Eigen::VectorXd& gradient_vector);

         private:

            using SolverType = Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>;

            std::shared_ptr<SolverType> solver() { return hessian_solver_; }  

            //Implements one iteration of the Gauss-Newton algorithm 
            bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

            //Shared pointer for the Hessian factorization solver
            std::shared_ptr<SolverType> hessian_solver_ = std::make_shared<SolverType>();

            //Flag indicating whether the Hessian sparsity pattern is initialized
            bool pattern_initialized_ = false;

            //Solver configuration parameters
            const Params params_;

            //Allows `Covariance` class to access internal solver object
            friend class Covariance;
    };
} // namespace finaleicp