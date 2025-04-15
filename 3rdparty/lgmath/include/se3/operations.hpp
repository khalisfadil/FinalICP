#pragma once

#include <Eigen/Core>

namespace math{
    namespace se3{

        //Constructs the 4x4 skew-symmetric matrix (hat operator) from translation and axis-angle vectors. 1
        Eigen::Matrix4d hat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis);

        //Constructs the 4x4 skew-symmetric matrix (hat operator) from a 6x1 SE(3) algebra vector. 2
        Eigen::Matrix4d hat(const Eigen::Matrix<double, 6, 1>& xi);

        //Constructs the 6x6 curly-hat matrix from translation and axis-angle vectors. 3
        Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Vector3d& rho, const Eigen::Vector3d& aaxis);

        //Constructs the 6x6 curly-hat matrix from a 6x1 SE(3) algebra vector. 4
        Eigen::Matrix<double, 6, 6> curlyhat(const Eigen::Matrix<double, 6, 1>& xi);

        //Converts a 3D point into a 4x6 matrix (circle-dot operator). 5
        Eigen::Matrix<double, 4, 6> point2fs(const Eigen::Vector3d& p, double scale = 1.0);

        //Converts a 3D point into a 6x4 matrix (double-circle operator). 6
        Eigen::Matrix<double, 6, 4> point2sf(const Eigen::Vector3d& p, double scale = 1.0);

        //Computes the SE(3) transformation analytically from translation and axis-angle vectors. 7
        void vec2tran_analytical(const Eigen::Vector3d& rho_ba, const Eigen::Vector3d& aaxis_ba, Eigen::Matrix3d* out_C_ab, Eigen::Vector3d* out_r_ba_ina);

        //Computes the SE(3) transformation numerically from translation and axis-angle vectors. 8
        void vec2tran_numerical(const Eigen::Vector3d& rho_ba, const Eigen::Vector3d& aaxis_ba, Eigen::Matrix3d* out_C_ab, Eigen::Vector3d* out_r_ba_ina, unsigned int numTerms);

        //Computes the SE(3) transformation from a 6x1 SE(3) algebra vector. 9
        void vec2tran(const Eigen::Matrix<double, 6, 1>& xi_ba, Eigen::Matrix3d* out_C_ab, Eigen::Vector3d* out_r_ba_ina, unsigned int numTerms = 0);

        //Computes the 4x4 SE(3) transformation matrix from a 6x1 SE(3) algebra vector. 10
        Eigen::Matrix4d vec2tran(const Eigen::Matrix<double, 6, 1>& xi_ba, unsigned int numTerms = 0);

        //Computes the logarithmic map from rotation and translation to a 6x1 SE(3) algebra vector. 11
        Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Matrix3d& C_ab, const Eigen::Vector3d& r_ba_ina);

        //Computes the logarithmic map from a 4x4 SE(3) transformation matrix. 12
        Eigen::Matrix<double, 6, 1> tran2vec(const Eigen::Matrix4d& T_ab);

        //Computes the 6x6 adjoint transformation matrix from rotation and translation.13
        Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Matrix3d& C_ab, const Eigen::Vector3d& r_ba_ina);

        //Computes the 6x6 adjoint transformation matrix from a 4x4 transformation matrix. 14
        Eigen::Matrix<double, 6, 6> tranAd(const Eigen::Matrix4d& T_ab);

        //Computes the 3x3 Q matrix for SE(3) from translation and axis-angle vectors. 15
        Eigen::Matrix3d vec2Q(const Eigen::Vector3d& rho_ba, const Eigen::Vector3d& aaxis_ba);

        //Computes the 3x3 Q matrix for SE(3) from a 6x1 SE(3) algebra vector. 16
        Eigen::Matrix3d vec2Q(const Eigen::Matrix<double, 6, 1>& xi_ba);

        //Computes the 6x6 left Jacobian of SE(3) from translation and axis-angle vectors. 17
        Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Vector3d& rho_ba, const Eigen::Vector3d& aaxis_ba);

        //Computes the 6x6 left Jacobian of SE(3) from a 6x1 SE(3) algebra vector. 18
        Eigen::Matrix<double, 6, 6> vec2jac(const Eigen::Matrix<double, 6, 1>& xi_ba, unsigned int numTerms = 0);

        //Computes the 6x6 inverse left Jacobian of SE(3) from translation and axis-angle vectors. 19
        Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Vector3d& rho_ba, const Eigen::Vector3d& aaxis_ba);

        //Computes the 6x6 inverse left Jacobian of SE(3) from a 6x1 SE(3) algebra vector. 20
        Eigen::Matrix<double, 6, 6> vec2jacinv(const Eigen::Matrix<double, 6, 1>& xi_ba, unsigned int numTerms = 0);

    }   // namespace se3
}   // namespace math