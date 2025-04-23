#pragma once

#include <solver/gausnewtonsolver.hpp>

namespace finalicp {

    //Implements the Gauss-Newton optimization algorithm.
    class LevMarqGaussNewtonSolver : public GaussNewtonSolver {

        public:
            struct Params : public GaussNewtonSolver::Params {
                double ratio_threshold = 0.25;                  //Minimum ratio of actual to predicted cost reduction, shrink trust region if lower (range: 0.0-1.0)
                double shrink_coeff = 0.1;                      //Amount to shrink by (range: <1.0)
                double grow_coeff = 10.0;                        //Amount to grow by (range: >1.0)
                unsigned int max_shrink_steps = 50;             //Maximum number of times to shrink trust region before giving up
            };
            
            LevMarqGaussNewtonSolver(Problem& problem, const Params& params);

         private:

            //Performs the linearization, solves the Gauss-Newton system, and updates the state.
            bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

            //Computes the solveLevMarq
            Eigen::VectorXd solveLevMarq(Eigen::SparseMatrix<double>& approximate_hessian, const Eigen::VectorXd& gradient_vector, double diagonal_coeff);

            //Computes the predicted cost reduction.
            double predictedReduction(const Eigen::SparseMatrix<double>& approximate_hessian, const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step);

            double diag_coeff_ = 1e-7;                //Trust region size

            const Params params_;
    };
} // namespace finaleicp