#pragma once

#include <problem/lossfunc/baselossfunc.hpp>

#include <cmath>

namespace finalicp {
    class HuberLossFunc : public BaseLossFunc {
        public:
            using Ptr = std::shared_ptr<HuberLossFunc>;
            using ConstPtr = std::shared_ptr<const HuberLossFunc>;

            //Factory method to create a shared instance of CauchyLossFunc.
            static Ptr MakeShared(double k) { return std::make_shared<HuberLossFunc>(k); }

            //Constructor to initialize Cauchy loss function.
            HuberLossFunc(double k) : k_(k) {}

            //Computes the cost function value.
            double cost(double whitened_error_norm) const override {
                double e2 = whitened_error_norm * whitened_error_norm;
                double abse = fabs(whitened_error_norm);
                if (abse <= k_) {
                return 0.5 * e2;
                } else {
                return k_ * (abse - 0.5 * k_);
                }
            }

            //Computes the weight for iteratively reweighted least squares (IRLS).
            double weight(double whitened_error_norm) const override {
                double abse = fabs(whitened_error_norm);
                if (abse <= k_) {
                return 1.0;
                } else {
                return k_ / abse;
                }
            }
        
        private:
            double k_;      //Cauchy constant
        
    };

}  // namespace finalicp