#pragma once

#include <problem/lossfunc/baselossfunc.hpp>

#include <cmath>

namespace finalicp {
    class CauchyLossFunc : public BaseLossFunc {
        public:
            using Ptr = std::shared_ptr<CauchyLossFunc>;
            using ConstPtr = std::shared_ptr<const CauchyLossFunc>;

            //Factory method to create a shared instance of CauchyLossFunc.
            static Ptr MakeShared(double k) {
                return std::make_shared<CauchyLossFunc>(k);
            }
            //Constructor to initialize Cauchy loss function.
            CauchyLossFunc(double k) : k_(k) {}

            //Computes the cost function value.
            double cost(double whitened_error_norm) const override {
                double e_div_k = fabs(whitened_error_norm) / k_;
                return 0.5 * k_ * k_ * std::log(1.0 + e_div_k * e_div_k);
            }

            //Computes the weight for iteratively reweighted least squares (IRLS).
            double weight(double whitened_error_norm) const override {
                double e_div_k = fabs(whitened_error_norm) / k_;
                return 1.0 / (1.0 + e_div_k * e_div_k);
            }
        
        private:
            double k_;      //Cauchy constant
        
    };

}  // namespace finalicp