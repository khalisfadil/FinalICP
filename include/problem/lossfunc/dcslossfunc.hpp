#pragma once

#include <problem/lossfunc/baselossfunc.hpp>

#include <cmath>

namespace finalicp {
    class DcsLossFunc : public BaseLossFunc {
        public:
            using Ptr = std::shared_ptr<DcsLossFunc>;
            using ConstPtr = std::shared_ptr<const DcsLossFunc>;

            //Factory method to create a shared instance of CauchyLossFunc.
            static Ptr MakeShared(double k) { return std::make_shared<DcsLossFunc>(k); }

            //Constructor to initialize Cauchy loss function.
            DcsLossFunc(double k) : k2_(k * k) {}

            //Computes the cost function value.
            double cost(double whitened_error_norm) const override {
                double e2 = whitened_error_norm * whitened_error_norm;
                if (e2 <= k2_) {
                return 0.5 * e2;
                } else {
                return 2.0 * k2_ * e2 / (k2_ + e2) - 0.5 * k2_;
                }
            }

            //Computes the weight for iteratively reweighted least squares (IRLS).
            double weight(double whitened_error_norm) const override {
                double e2 = whitened_error_norm * whitened_error_norm;
                if (e2 <= k2_) {
                return 1.0;
                } else {
                double k2e2 = k2_ + e2;
                return 4.0 * k2_ * k2_ / (k2e2 * k2e2);
                }
            }
        
        private:
            double k2_;      //Cauchy constant
        
    };

}  // namespace finalicp