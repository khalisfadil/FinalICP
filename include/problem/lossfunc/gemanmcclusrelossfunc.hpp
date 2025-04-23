#pragma once

#include <problem/lossfunc/baselossfunc.hpp>

#include <cmath>

namespace finalicp {
    class GemanMcClureLossFunc : public BaseLossFunc {
        public:
            using Ptr = std::shared_ptr<GemanMcClureLossFunc>;
            using ConstPtr = std::shared_ptr<const GemanMcClureLossFunc>;

            //Factory method to create a shared instance of CauchyLossFunc.
            static Ptr MakeShared(double k) {return std::make_shared<GemanMcClureLossFunc>(k);}

            //Constructor to initialize Cauchy loss function.
            GemanMcClureLossFunc(double k) : k2_(k * k) {}

            //Computes the cost function value.
            double cost(double whitened_error_norm) const override {
                double e2 = whitened_error_norm * whitened_error_norm;
                return 0.5 * e2 / (k2_ + e2);
            }

            //Computes the weight for iteratively reweighted least squares (IRLS).
            double weight(double whitened_error_norm) const override {
                double e2 = whitened_error_norm * whitened_error_norm;
                double k2e2 = k2_ + e2;
                return k2_ * k2_ / (k2e2 * k2e2);
            }
        
        private:
            double k2_;      //Cauchy constant
        
    };

}  // namespace finalicp