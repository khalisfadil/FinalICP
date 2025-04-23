#pragma once

#include <solver/gausnewtonsolver.hpp>

namespace finalicp {

    //Implements the Gauss-Newton optimization algorithm.
    class LineSearchGaussNewtonSolver : public GaussNewtonSolver {

        public:
            struct Params : public GaussNewtonSolver::Params {
                double backtrack_multiplier = 0.5;                  //Amount to decrease step after each backtrack
                unsigned int max_backtrack_steps = 10;              //Maximimum number of times to backtrack before giving up
            };
            
            LineSearchGaussNewtonSolver(Problem& problem, const Params& params);

         private:

            //Performs the linearization, solves the Gauss-Newton system, and updates the state.
            bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

            const Params params_;
    };
} // namespace finaleicp