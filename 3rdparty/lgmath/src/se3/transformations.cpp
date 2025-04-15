#include <se3/transformations.hpp>

#include <iostream>
#include <stdexcept>

#include <se3/operations.hpp>
#include <so3/operations.hpp>

namespace math {
    namespace se3 {

        // Default constructor (Identity transformation)
        Transformation::Transformation()
            : C_ba_(Eigen::Matrix3d::Identity()), r_ab_inb_(Eigen::Vector3d::Zero()) {}

        // Constructor from a 4x4 transformation matrix
        Transformation::Transformation(const Eigen::Matrix4d& T)
            : C_ba_(T.topLeftCorner<3, 3>()), r_ab_inb_(T.topRightCorner<3, 1>()) {
            this->reproject(false);
        }

        // Constructor from rotation and translation
        Transformation::Transformation(const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina)
            : C_ba_(C_ba), r_ab_inb_(Eigen::Vector3d::Zero()) { // Initialize to zero
            this->reproject(false);
            r_ab_inb_ = -C_ba_ * r_ba_ina; // Compute translation after reprojection
        }

        // Constructor from se(3) algebra vector
        Transformation::Transformation(const Eigen::Matrix<double, 6, 1>& xi_ab, unsigned int numTerms) {
            se3::vec2tran(xi_ab, &C_ba_, &r_ab_inb_, numTerms);
        }

        // Constructor from a dynamic-sized vector (must be 6x1)
        Transformation::Transformation(const Eigen::VectorXd& xi_ab) {
            if (xi_ab.rows() != 6) {
                throw std::invalid_argument("xi_ab must be a 6x1 vector.");
            }
            se3::vec2tran(xi_ab, &C_ba_, &r_ab_inb_, 0);
        }

        // Get the 4x4 transformation matrix representation
        Eigen::Matrix4d Transformation::matrix() const {
            Eigen::Matrix4d T_ba = Eigen::Matrix4d::Identity();
            T_ba.topLeftCorner<3, 3>() = C_ba_;
            T_ba.topRightCorner<3, 1>() = r_ab_inb_;
            return T_ba;
        }

        // Get rotation matrix
        const Eigen::Matrix3d& Transformation::C_ba() const {
            return C_ba_;
        }

        // Get translation vector in frame b
        const Eigen::Vector3d& Transformation::r_ab_inb() const {
            return r_ab_inb_;
        }

        // Get translation vector in frame a
        Eigen::Vector3d Transformation::r_ba_ina() const {
            return (-C_ba_.transpose() * r_ab_inb_).eval();
        }

        // Get the corresponding se(3) Lie algebra vector
        Eigen::Matrix<double, 6, 1> Transformation::vec() const {
            return se3::tran2vec(C_ba_, r_ab_inb_);
        }

        // Compute and return the inverse transformation
        Transformation Transformation::inverse() const {
            Transformation temp;
            temp.C_ba_ = C_ba_.transpose();
            temp.r_ab_inb_ = -temp.C_ba_ * r_ab_inb_;
            temp.reproject(false);
            return temp;
        }

        // Compute the 6x6 adjoint transformation matrix
        Eigen::Matrix<double, 6, 6> Transformation::adjoint() const {
            return se3::tranAd(C_ba_, r_ab_inb_);
        }

        // Ensures the transformation remains within SE(3)
        void Transformation::reproject(bool force) {
            bool isValidRotation = std::fabs(1.0 - C_ba_.determinant()) < 1e-6 &&
                                (C_ba_.transpose() * C_ba_ - Eigen::Matrix3d::Identity()).norm() < 1e-6;
            if (force || !isValidRotation) {
                C_ba_ = so3::vec2rot(so3::rot2vec(C_ba_));
            }
        }

        // In-place right-hand multiplication
        Transformation& Transformation::operator*=(const Transformation& T_rhs) {
            r_ab_inb_ += C_ba_ * T_rhs.r_ab_inb_;
            C_ba_ *= T_rhs.C_ba_;
            this->reproject(false);
            return *this;
        }

        // Right-hand multiplication (returns new transformation)
        Transformation Transformation::operator*(const Transformation& T_rhs) const {
            Transformation temp(*this);
            temp *= T_rhs;
            return temp;
        }

        // In-place right-hand multiplication with the inverse of T_rhs
        Transformation& Transformation::operator/=(const Transformation& T_rhs) {
            C_ba_ *= T_rhs.C_ba_.transpose();
            r_ab_inb_ += (-C_ba_ * T_rhs.r_ab_inb_);
            this->reproject(false);
            return *this;
        }

        // Right-hand multiplication with the inverse of T_rhs
        Transformation Transformation::operator/(const Transformation& T_rhs) const {
            Transformation temp(*this);
            temp /= T_rhs;
            return temp;
        }

        // Apply transformation to a 4D homogeneous vector
        Eigen::Vector4d Transformation::operator*(const Eigen::Ref<const Eigen::Vector4d>& p_a) const {
            Eigen::Vector4d p_b;
            p_b.head<3>() = C_ba_ * p_a.head<3>() + r_ab_inb_ * p_a[3];
            p_b[3] = p_a[3];
            return p_b;
        }

    } // namespace se3
} // namespace math

// Print transformation matrix
std::ostream& operator<<(std::ostream& out, const math::se3::Transformation& T) {
    out << "\n" << T.matrix() << "\n";
    return out;
}