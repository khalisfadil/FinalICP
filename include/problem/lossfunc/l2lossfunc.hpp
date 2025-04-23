#pragma once

#include <problem/lossfunc/baselossfunc.hpp>

#include <cmath>

namespace finalicp {
    class L2LossFunc : public BaseLossFunc {
        public:
            using Ptr = std::shared_ptr<L2LossFunc>;
            using ConstPtr = std::shared_ptr<const L2LossFunc>;

            //Factory method to create a shared instance of CauchyLossFunc.
            static Ptr MakeShared() { return std::make_shared<L2LossFunc>(); }

            //Constructor to initialize Cauchy loss function.
            L2LossFunc() = default;

            //Computes the cost function value.
            double cost(double whitened_error_norm) const override {
                return 0.5 * whitened_error_norm * whitened_error_norm;
            }

            //Computes the weight for iteratively reweighted least squares (IRLS).
            double weight(double whitened_error_norm) const override { return 1.0; }
    };

}  // namespace finalicp