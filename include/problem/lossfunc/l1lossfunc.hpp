#pragma once

#include <problem/lossfunc/baselossfunc.hpp>

#include <cmath>

namespace finalicp {
    class L1LossFunc : public BaseLossFunc {
        public:
            using Ptr = std::shared_ptr<L1LossFunc>;
            using ConstPtr = std::shared_ptr<const L1LossFunc>;

            //Factory method to create a shared instance of CauchyLossFunc.
            static Ptr MakeShared() { return std::make_shared<L1LossFunc>(); }

            //Constructor to initialize Cauchy loss function.
            L1LossFunc() = default;

            //Computes the cost function value.
            double cost(double whitened_error_norm) const override {
                return fabs(whitened_error_norm);
            }

            //Computes the weight for iteratively reweighted least squares (IRLS).
            double weight(double whitened_error_norm) const override {
                return 1.0 / fabs(whitened_error_norm);
            }
        
    };

}  // namespace finalicp