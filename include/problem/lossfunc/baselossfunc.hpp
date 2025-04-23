#pragma once

#include <memory>

namespace finalicp {
    class BaseLossFunc {
        public:
            using Ptr = std::shared_ptr<BaseLossFunc>;
            using ConstPtr = std::shared_ptr<const BaseLossFunc>;

            //Default virtual destructor
            virtual ~BaseLossFunc() = default;

            //Computes the cost of a given whitened error norm.
            virtual double cost(double whitened_error_norm) const = 0;

            //Computes the weight for iteratively reweighted least squares (IRLS).
            virtual double weight(double whitened_error_norm) const = 0;
    };

}  // namespace finalicp