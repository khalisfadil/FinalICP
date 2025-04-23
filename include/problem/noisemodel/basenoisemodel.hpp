#pragma once

#include <Eigen/Dense>


namespace finalicp {
    //Enumeration of ways to define noise models.
    enum class NoiseType { COVARIANCE, INFORMATION, SQRT_INFORMATION };

    template <int DIM>
    class BaseNoiseModel {
        public:
            using Ptr = std::shared_ptr<BaseNoiseModel<DIM>>;
            using ConstPtr = std::shared_ptr<const BaseNoiseModel<DIM>>;

            using MatrixT = Eigen::Matrix<double, DIM, DIM>;
            using VectorT = Eigen::Matrix<double, DIM, 1>;

            //Virtual destructor
            virtual ~BaseNoiseModel() = default;

            //Get a reference to the square root information matrix
            virtual MatrixT getSqrtInformation() const = 0;

            //Compute the whitened error norm, defined as:
            virtual double getWhitenedErrorNorm(const VectorT& rawError) const = 0;

            //Compute the whitened error vector:
            virtual VectorT whitenError(const VectorT& rawError) const = 0;

        
    };

}  // namespace finalicp
