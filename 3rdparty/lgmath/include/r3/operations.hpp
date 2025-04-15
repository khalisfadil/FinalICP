#pragma once

#include <stdexcept>

#include <Eigen/Dense>
#include <r3/types.hpp>
#include <se3/operations.hpp>
#include <se3/transformationswithcovariance.hpp>

namespace math{
    namespace r3{

        //The transform covariance is required to be set
        static constexpr bool COVARIANCE_REQUIRED = true;

        //The transform covariance is not required to be set
        static constexpr bool COVARIANCE_NOT_REQUIRED = false;

        //Transforms a 3D point covariance using a **certain** transformation (SE(3)).
        template <bool THROW_IF_UNSET = COVARIANCE_REQUIRED>
            CovarianceMatrix transformCovariance(const math::se3::Transformation &T_ba,
                                                const CovarianceMatrixConstRef &cov_a,
                                                const HPointConstRef &p_b = HPoint()) {
                (void)&p_b;  // Unused for certain transforms
                static_assert(!THROW_IF_UNSET,
                                "Error: Transformation never has covariance explicitly set.");

                // The covariance is transformed by the rotation matrix
                return T_ba.C_ba() * cov_a * T_ba.C_ba().transpose();
            }

        //Transforms a 3D point covariance using an **uncertain** transformation.
        template <bool THROW_IF_UNSET = COVARIANCE_REQUIRED>
            CovarianceMatrix transformCovariance(
                const math::se3::TransformationWithCovariance &T_ba,
                const CovarianceMatrixConstRef &cov_a, const HPointConstRef &p_b) {
                // Ensure the transform has covariance, if required
                if (THROW_IF_UNSET && !T_ba.covarianceSet()) {
                    throw std::runtime_error(
                        "Error: TransformationWithCovariance does not have covariance set.");
                }

                // Transform covariance using base transformation (rotation matrix only)
                const auto &T_ba_base = static_cast<const math::se3::Transformation &>(T_ba);
                CovarianceMatrix cov_b = transformCovariance<false>(T_ba_base, cov_a, p_b);

                // Add uncertainty from the transformation itself
                if (T_ba.covarianceSet()) {
                    auto jacobian = math::se3::point2fs(p_b.hnormalized()).topRows<3>(); // Compute Jacobian
                    cov_b += jacobian * T_ba.cov() * jacobian.transpose();
                }

                return cov_b;
            }

    }   // namespace r3
}   // namespace math