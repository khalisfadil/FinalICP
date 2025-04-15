#include <so3/operations.hpp>

#include <stdio.h>
#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

namespace math {
    namespace so3 {
        // -----------------------------------------------------------------------------
        // SO(3) Hat Operator
        // -----------------------------------------------------------------------------

        Eigen::Matrix3d hat(const Eigen::Vector3d& vector) {
            Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
            mat(0, 1) = -vector[2];
            mat(0, 2) = vector[1];
            mat(1, 0) = vector[2];
            mat(1, 2) = -vector[0];
            mat(2, 0) = -vector[1];
            mat(2, 1) = vector[0];
            return mat;
        }

        // -----------------------------------------------------------------------------
        // SO(3) Exponential Map (Vector to Rotation)
        // -----------------------------------------------------------------------------

        Eigen::Matrix3d vec2rot(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms) {
            const double phi_ba = aaxis_ba.norm();
            if (phi_ba < 1e-12) {
                return Eigen::Matrix3d::Identity();
            }
            if (numTerms == 0) {
                Eigen::Vector3d axis = aaxis_ba / phi_ba;
                const double sinphi_ba = std::sin(phi_ba);
                const double cosphi_ba = std::cos(phi_ba);
                Eigen::Matrix3d C_ab = cosphi_ba * Eigen::Matrix3d::Identity() +
                                    (1.0 - cosphi_ba) * axis * axis.transpose() +
                                    sinphi_ba * math::so3::hat(axis);
                Eigen::AngleAxisd aa(phi_ba, axis);
                Eigen::Matrix3d C_aa = aa.toRotationMatrix();
                if (!C_ab.isApprox(C_aa, 1e-6)) {
                    return C_aa;
                }
                return C_ab;
            } else {
                Eigen::Matrix3d C_ab = Eigen::Matrix3d::Identity();
                Eigen::Matrix3d x_small = math::so3::hat(aaxis_ba);
                Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();
                for (unsigned int n = 1; n <= numTerms; ++n) {
                    x_small_n = x_small_n * x_small / static_cast<double>(n);
                    C_ab += x_small_n;
                }
                return C_ab;
            }
        }

        void vec2rot(const Eigen::Vector3d& aaxis_ba,
                    Eigen::Matrix3d* out_C_ab,
                    Eigen::Matrix3d* out_J_ab) {
            assert(out_C_ab && "Null pointer out_C_ab in vec2rot");
            assert(out_J_ab && "Null pointer out_J_ab in vec2rot");

            *out_J_ab = vec2jac(aaxis_ba);
            *out_C_ab = Eigen::Matrix3d::Identity() + hat(aaxis_ba) * (*out_J_ab);
        }

        // -----------------------------------------------------------------------------
        // SO(3) Logarithmic Map (Rotation to Vector)
        // -----------------------------------------------------------------------------

        Eigen::Vector3d rot2vec(const Eigen::Matrix3d& C_ab, double eps) {
            // Get angle using trace
            const double trace_term = 0.5 * (C_ab.trace() - 1.0);
            const double phi_ba = std::acos(std::clamp(trace_term, -1.0, 1.0));
            const double sinphi_ba = std::sin(phi_ba);

            if (std::fabs(sinphi_ba) > eps) {
                // General case: angle not near 0 or π
                Eigen::Vector3d axis;
                axis << C_ab(2, 1) - C_ab(1, 2),
                        C_ab(0, 2) - C_ab(2, 0),
                        C_ab(1, 0) - C_ab(0, 1);
                return (0.5 * phi_ba / sinphi_ba) * axis;
            } else if (std::fabs(phi_ba) > eps) {
                // Angle near π: use AngleAxisd for robust axis computation
                Eigen::AngleAxisd aa(C_ab);
                double theta = aa.angle();
                Eigen::Vector3d axis = aa.axis();
                if (theta > M_PI) {
                    theta = 2.0 * M_PI - theta;  // Normalize to [0, π]
                    axis = -axis;
                }
                if (!C_ab.isApprox(Eigen::Matrix3d::Identity(), 1e-6)) {
                    return theta * axis;
                } else {
                    throw std::runtime_error(
                        "so3 logarithmic map failed: rotation matrix is identity but phi_ba > eps");
                }
            } else {
                // Angle near zero
                return Eigen::Vector3d::Zero();
            }
        }

        // -----------------------------------------------------------------------------
        // SO(3) Left Jacobian
        // -----------------------------------------------------------------------------

        Eigen::Matrix3d vec2jac(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms) {
            const double phi_ba = aaxis_ba.norm();
            if (phi_ba < 1e-12) {
                return Eigen::Matrix3d::Identity();
            }

            if (numTerms == 0) {
                const double sinphi = std::sin(phi_ba);
                const double cosphi = std::cos(phi_ba);
                const double sinTerm = sinphi / phi_ba;
                const double cosTerm = (1.0 - cosphi) / phi_ba;
                Eigen::Vector3d axis = aaxis_ba / phi_ba;
                Eigen::Matrix3d axis_outer = axis * axis.transpose();
                return sinTerm * Eigen::Matrix3d::Identity() +
                    (1.0 - sinTerm) * axis_outer +
                    cosTerm * hat(axis);
            }

            Eigen::Matrix3d J_ab = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d x_small = hat(aaxis_ba);
            Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();
            for (unsigned int n = 1; n <= numTerms; ++n) {
                x_small_n = x_small_n * x_small / static_cast<double>(n + 1);
                J_ab += x_small_n;
            }
            return J_ab;
        }

        // -----------------------------------------------------------------------------
        // SO(3) Inverse Left Jacobian
        // -----------------------------------------------------------------------------

        Eigen::Matrix3d vec2jacinv(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms) {
            const double phi_ba = aaxis_ba.norm();
            if (phi_ba < 1e-12) {
                return Eigen::Matrix3d::Identity();
            }

            if (numTerms == 0) {
                const double halfphi = 0.5 * phi_ba;
                const double cotanTerm = halfphi / std::tan(halfphi);
                Eigen::Vector3d axis = aaxis_ba / phi_ba;
                Eigen::Matrix3d axis_outer = axis * axis.transpose();
                return cotanTerm * Eigen::Matrix3d::Identity() +
                    (1.0 - cotanTerm) * axis_outer -
                    halfphi * hat(axis);
            }

            if (numTerms > 20) {
                //std::cerr << "Numerical vec2jacinv: numTerms > 20 not supported, returning identity" << std::endl;
                return Eigen::Matrix3d::Identity();
            }

            static const double bernoulli[] = {1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
                                            -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0, 0.0,
                                            -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0};

            Eigen::Matrix3d J_ab_inv = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d x_small = hat(aaxis_ba);
            Eigen::Matrix3d x_small_n = Eigen::Matrix3d::Identity();
            for (unsigned int n = 1; n <= numTerms; ++n) {
                x_small_n = x_small_n * x_small / static_cast<double>(n);
                J_ab_inv += bernoulli[n] * x_small_n;
            }
            return J_ab_inv;
        }
    }   // namespace se3
}   // namespace math

