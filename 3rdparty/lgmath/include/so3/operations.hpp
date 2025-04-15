#pragma once

#include <Eigen/Core>

namespace math{
    namespace so3{

            //Constructs a 3x3 skew-symmetric matrix (hat operator) from a 3x1 vector.
            Eigen::Matrix3d hat(const Eigen::Vector3d& vector);                                                  

            //Computes a 3x3 rotation matrix from an axis-angle vector using the exponential map.
            Eigen::Matrix3d vec2rot(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms = 0);  

            //Computes a 3x3 rotation matrix from an axis-angle vector using the exponential map.
            void vec2rot(const Eigen::Vector3d& aaxis_ba, Eigen::Matrix3d* out_C_ab, Eigen::Matrix3d* out_J_ab);

            //Computes the axis-angle vector from a 3x3 rotation matrix using the logarithmic map.
            Eigen::Vector3d rot2vec(const Eigen::Matrix3d& C_ab, double eps = 1e-6);

            //Computes the 3x3 left Jacobian of SO(3) from an axis-angle vector.
            Eigen::Matrix3d vec2jac(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms = 0);

            //Computes the 3x3 inverse left Jacobian of SO(3) from an axis-angle vector.
            Eigen::Matrix3d vec2jacinv(const Eigen::Vector3d& aaxis_ba, unsigned int numTerms = 0);

    }   // namespace so3
}   // namespace math