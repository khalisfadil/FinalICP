#include <se3/operations.hpp>

#include <stdio.h>
#include <stdexcept>

#include <Eigen/Dense>

#include <so3/operations.hpp>

namespace math {
    namespace se3 {

        // -----------------------------------------------------------------------------
        // SE(3) Hat Operators
        // -----------------------------------------------------------------------------

        Eigen::Matrix4d hat(const Eigen::Vector3d& rho,
                            const Eigen::Vector3d& aaxis) {
            Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
            mat.topLeftCorner<3, 3>() = so3::hat(aaxis);
            mat.topRightCorner<3, 1>() = rho;
            return mat;
        }

        Eigen::Matrix4d hat(const Eigen::Matrix<double, 6, 1>& xi) {
            return hat(xi.head<3>(), xi.tail<3>());
        }

        // -----------------------------------------------------------------------------
        // SE(3) Curly Hat Operators
        // -----------------------------------------------------------------------------

        Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Vector3d& rho,
                                            const Eigen::Vector3d& aaxis) {
            Eigen::Matrix<double, 6, 6> mat = Eigen::Matrix<double, 6, 6>::Zero();
            mat.topLeftCorner<3, 3>() = mat.bottomRightCorner<3, 3>() = so3::hat(aaxis);
            mat.topRightCorner<3, 3>() = so3::hat(rho);
            return mat;
        }

        Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Matrix<double, 6, 1>& xi) {
            return curlyhat(xi.head<3>(), xi.tail<3>());
        }

        // -----------------------------------------------------------------------------
        // Point Transformations to Lie Algebra
        // -----------------------------------------------------------------------------

        Eigen::Matrix<double, 4, 6> point2fs(const Eigen::Vector3d& p, double scale) {
            Eigen::Matrix<double, 4, 6> mat = Eigen::Matrix<double, 4, 6>::Zero();
            mat.topLeftCorner<3, 3>() = scale * Eigen::Matrix3d::Identity();
            mat.topRightCorner<3, 3>() = -so3::hat(p);
            return mat;
        }

        Eigen::Matrix<double, 6, 4> point2sf(const Eigen::Vector3d& p, double scale) {
            Eigen::Matrix<double, 6, 4> mat = Eigen::Matrix<double, 6, 4>::Zero();
            mat.bottomLeftCorner<3, 3>() = -so3::hat(p);
            mat.topRightCorner<3, 1>() = p;  // No scaling
            return mat;
        }

        // -----------------------------------------------------------------------------
        // SE(3) Exponential Map (Vector to Transformation)
        // -----------------------------------------------------------------------------

        void vec2tran_analytical(const Eigen::Vector3d& rho_ba,
                                const Eigen::Vector3d& aaxis_ba,
                                Eigen::Matrix3d* out_C_ab,
                                Eigen::Vector3d* out_r_ba_ina) {
            assert(out_C_ab && "Null pointer out_C_ab in vec2tran_analytical");
            assert(out_r_ba_ina && "Null pointer out_r_ba_ina in vec2tran_analytical");

            if (aaxis_ba.squaredNorm() < 1e-24) {
                *out_C_ab = Eigen::Matrix3d::Identity();
                *out_r_ba_ina = rho_ba;
                return;
            }

            Eigen::Matrix3d J_ab;
            so3::vec2rot(aaxis_ba, out_C_ab, &J_ab);
            *out_r_ba_ina = J_ab * rho_ba;
        }

        void vec2tran_numerical(const Eigen::Vector3d& rho_ba,
                                const Eigen::Vector3d& aaxis_ba,
                                Eigen::Matrix3d* out_C_ab,
                                Eigen::Vector3d* out_r_ba_ina,
                                unsigned int numTerms) {
            assert(out_C_ab && "Null pointer out_C_ab in vec2tran_numerical");
            assert(out_r_ba_ina && "Null pointer out_r_ba_ina in vec2tran_numerical");

            Eigen::Matrix4d T_ab = Eigen::Matrix4d::Identity();
            Eigen::Matrix<double, 6, 1> xi_ba;
            xi_ba.head<3>().noalias() = rho_ba;  // Scalar assignment for safety
            xi_ba.tail<3>().noalias() = aaxis_ba;  // Scalar assignment for safety

            Eigen::Matrix4d x_small = hat(xi_ba);
            Eigen::Matrix4d x_small_n = Eigen::Matrix4d::Identity();

            for (unsigned int n = 1; n <= numTerms; ++n) {
                x_small_n = x_small_n * x_small / static_cast<double>(n);
                T_ab += x_small_n;
            }

            *out_C_ab = T_ab.topLeftCorner<3, 3>();
            *out_r_ba_ina = T_ab.topRightCorner<3, 1>();
        }

        void vec2tran(const Eigen::Matrix<double, 6, 1>& xi_ba,
                    Eigen::Matrix3d* out_C_ab,
                    Eigen::Vector3d* out_r_ba_ina,
                    unsigned int numTerms) {
            assert(out_C_ab && "Null pointer out_C_ab in vec2tran");
            assert(out_r_ba_ina && "Null pointer out_r_ba_ina in vec2tran");

            if (numTerms == 0) {
                vec2tran_analytical(xi_ba.head<3>(), xi_ba.tail<3>(), out_C_ab, out_r_ba_ina);
            } else {
                vec2tran_numerical(xi_ba.head<3>(), xi_ba.tail<3>(), out_C_ab, out_r_ba_ina, numTerms);
            }
        }

        Eigen::Matrix4d vec2tran(const Eigen::Matrix<double, 6, 1>& xi_ba,
                                unsigned int numTerms) {
            Eigen::Matrix3d C_ab;
            Eigen::Vector3d r_ba_ina;
            vec2tran(xi_ba, &C_ab, &r_ba_ina, numTerms);
            Eigen::Matrix4d T_ab = Eigen::Matrix4d::Identity();
            T_ab.topLeftCorner<3, 3>() = C_ab;
            T_ab.topRightCorner<3, 1>() = r_ba_ina;
            return T_ab;
        }

        // -----------------------------------------------------------------------------
        // SE(3) Logarithmic Map (Transformation to Vector)
        // -----------------------------------------------------------------------------

        Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Matrix3d& C_ab,
                                            const Eigen::Vector3d& r_ba_ina) {
            Eigen::Matrix<double, 6, 1> xi_ba;
            Eigen::Vector3d aaxis_ba = so3::rot2vec(C_ab);
            Eigen::Vector3d rho_ba = so3::vec2jacinv(aaxis_ba) * r_ba_ina;
            xi_ba.head<3>().noalias() = rho_ba;  // Scalar assignment to avoid AVX over-read
            xi_ba.tail<3>().noalias() = aaxis_ba;  // Scalar assignment to avoid AVX over-read
            return xi_ba;
        }

        Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Matrix4d& T_ab) {
            return tran2vec(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
        }

        // -----------------------------------------------------------------------------
        // SE(3) Adjoint Transformation
        // -----------------------------------------------------------------------------

        Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Matrix3d& C_ab,
                                        const Eigen::Vector3d& r_ba_ina) {
            Eigen::Matrix<double, 6, 6> adjoint_T_ab = Eigen::Matrix<double, 6, 6>::Zero();
            adjoint_T_ab.topLeftCorner<3, 3>() = adjoint_T_ab.bottomRightCorner<3, 3>() = C_ab;
            adjoint_T_ab.topRightCorner<3, 3>() = so3::hat(r_ba_ina) * C_ab;
            return adjoint_T_ab;
        }

        Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Matrix4d& T_ab) {
            return tranAd(T_ab.topLeftCorner<3, 3>(), T_ab.topRightCorner<3, 1>());
        }

        // -----------------------------------------------------------------------------
        // SE(3) Q Matrix
        // -----------------------------------------------------------------------------

        Eigen::Matrix3d vec2Q(const Eigen::Vector3d& rho_ba,
                            const Eigen::Vector3d& aaxis_ba) {
            const double ang = aaxis_ba.norm();
            if (ang < 1e-12) return 0.5 * so3::hat(rho_ba);

            const double ang2 = ang * ang;
            const double ang3 = ang2 * ang;
            const double ang4 = ang3 * ang;
            const double ang5 = ang4 * ang;
            const double cang = std::cos(ang);
            const double sang = std::sin(ang);
            const double m2 = (ang - sang) / ang3;
            const double m3 = (1.0 - 0.5 * ang2 - cang) / ang4;
            const double m4 = 0.5 * (m3 - 3 * (ang - sang - ang3 / 6) / ang5);

            Eigen::Matrix3d rx = so3::hat(rho_ba);
            Eigen::Matrix3d px = so3::hat(aaxis_ba);
            return 0.5 * rx + m2 * (px * rx + rx * px + px * rx * px) -
                m3 * (px * px * rx + rx * px * px - 3 * px * rx * px) -
                m4 * (px * rx * px * px + px * px * rx * px);
        }

        Eigen::Matrix3d vec2Q(const Eigen::Matrix<double, 6, 1>& xi_ba) {
            return vec2Q(xi_ba.head<3>(), xi_ba.tail<3>());
        }

        // -----------------------------------------------------------------------------
        // SE(3) Left Jacobian
        // -----------------------------------------------------------------------------

        Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Vector3d& rho_ba,
                                            const Eigen::Vector3d& aaxis_ba) {
            Eigen::Matrix<double, 6, 6> J_ab = Eigen::Matrix<double, 6, 6>::Zero();
            if (aaxis_ba.norm() < 1e-12) {
                J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity();
                J_ab.topRightCorner<3, 3>() = 0.5 * so3::hat(rho_ba);
            } else {
                J_ab.topLeftCorner<3, 3>() = J_ab.bottomRightCorner<3, 3>() = so3::vec2jac(aaxis_ba);
                J_ab.topRightCorner<3, 3>() = vec2Q(rho_ba, aaxis_ba);
            }
            return J_ab;
        }

        Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Matrix<double, 6, 1>& xi_ba,
                                            unsigned int numTerms) {
            if (numTerms == 0) {
                return vec2jac(xi_ba.head<3>(), xi_ba.tail<3>());
            }

            Eigen::Matrix<double, 6, 6> J_ab = Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Matrix<double, 6, 6> x_small = curlyhat(xi_ba);
            Eigen::Matrix<double, 6, 6> x_small_n = Eigen::Matrix<double, 6, 6>::Identity();

            for (unsigned int n = 1; n <= numTerms; ++n) {
                x_small_n = x_small_n * x_small / static_cast<double>(n + 1);
                J_ab += x_small_n;
            }
            return J_ab;
        }

        // -----------------------------------------------------------------------------
        // SE(3) Inverse Left Jacobian
        // -----------------------------------------------------------------------------

        Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Vector3d& rho_ba,
                                            const Eigen::Vector3d& aaxis_ba) {
            Eigen::Matrix<double, 6, 6> J66_ab_inv = Eigen::Matrix<double, 6, 6>::Zero();
            if (aaxis_ba.norm() < 1e-12) {
                J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity();
                J66_ab_inv.topRightCorner<3, 3>() = -0.5 * so3::hat(rho_ba);
            } else {
                Eigen::Matrix3d J33_ab_inv = so3::vec2jacinv(aaxis_ba);
                J66_ab_inv.topLeftCorner<3, 3>() = J66_ab_inv.bottomRightCorner<3, 3>() = J33_ab_inv;
                J66_ab_inv.topRightCorner<3, 3>() = -J33_ab_inv * vec2Q(rho_ba, aaxis_ba) * J33_ab_inv;
            }
            return J66_ab_inv;
        }

        Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Matrix<double, 6, 1>& xi_ba,
                                            unsigned int numTerms) {
            if (numTerms == 0) {
                return vec2jacinv(xi_ba.head<3>(), xi_ba.tail<3>());
            }

            if (numTerms > 20) {
                //std::cerr << "Warning: vec2jacinv numTerms > 20 not supported, returning identity" << std::endl;
                return Eigen::Matrix<double, 6, 6>::Identity();
            }

            Eigen::Matrix<double, 6, 6> J_ab = Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Matrix<double, 6, 6> x_small = curlyhat(xi_ba);
            Eigen::Matrix<double, 6, 6> x_small_n = Eigen::Matrix<double, 6, 6>::Identity();

            // Precomputed Bernoulli numbers (up to 20 terms)
            static const double bernoulli[] = {1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
                                            -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0, 0.0,
                                            -3617.0 / 510.0, 0.0, 43867.0 / 798.0, 0.0, -174611.0 / 330.0};

            for (unsigned int n = 1; n <= numTerms; ++n) {
                x_small_n = x_small_n * x_small / static_cast<double>(n);
                J_ab += bernoulli[n] * x_small_n;
            }
            return J_ab;
        }
    }   // namespace se3
}   // namespace math