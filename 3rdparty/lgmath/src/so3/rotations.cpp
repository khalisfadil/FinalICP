#include <so3/rotations.hpp>

#include <stdexcept>

#include <so3/operations.hpp>

namespace math {
    namespace so3 {
        // -----------------------------------------------------------------------------
        // Default constructor (identity rotation)
        // -----------------------------------------------------------------------------

        Rotation::Rotation() : C_ba_(Eigen::Matrix3d::Identity()) {}

        // -----------------------------------------------------------------------------
        // Constructor from a 3x3 rotation matrix with optional reprojection
        // -----------------------------------------------------------------------------

        Rotation::Rotation(const Eigen::Matrix3d& C) : C_ba_(C) {
            this->reproject(false);
        }

        // -----------------------------------------------------------------------------
        // Constructor from an axis-angle vector (exponential map)
        // -----------------------------------------------------------------------------

        Rotation::Rotation(const Eigen::Vector3d& aaxis_ab, unsigned int numTerms) {
            C_ba_ = so3::vec2rot(aaxis_ab, numTerms);
        }

        // -----------------------------------------------------------------------------
        // Constructor from an Eigen vector (must be 3x1)
        // -----------------------------------------------------------------------------

        Rotation::Rotation(const Eigen::VectorXd& aaxis_ab) {
            if (aaxis_ab.rows() != 3) {
                throw std::invalid_argument("Rotation vector must be 3x1.");
            }
            C_ba_ = so3::vec2rot(aaxis_ab);
            }

        // -----------------------------------------------------------------------------
        // Get the underlying rotation matrix
        // -----------------------------------------------------------------------------

        const Eigen::Matrix3d& Rotation::matrix() const{
            return C_ba_;
        }

        // -----------------------------------------------------------------------------
        // Convert rotation matrix to axis-angle representation
        // -----------------------------------------------------------------------------

        Eigen::Vector3d Rotation::vec() const {
            return so3::rot2vec(C_ba_);
        }

        // -----------------------------------------------------------------------------
        // Compute the inverse (transpose) of the rotation matrix
        // -----------------------------------------------------------------------------

        Rotation Rotation::inverse() const {
            Rotation temp;
            temp.C_ba_ = C_ba_.transpose();
            temp.reproject(false);
            return temp;
        }

        // -----------------------------------------------------------------------------
        // Ensures the matrix is a valid rotation matrix (optional forced reprojection)
        // -----------------------------------------------------------------------------

        void Rotation::reproject(bool force) {
            if (force || fabs(1.0 - this->C_ba_.determinant()) > 1e-6) {
                C_ba_ = so3::vec2rot(so3::rot2vec(C_ba_));
            }
        }

        // -----------------------------------------------------------------------------
        // In-place right-hand multiplication with another rotation. 
        // -----------------------------------------------------------------------------

        Rotation& Rotation::operator*=(const Rotation& C_rhs) {
            C_ba_ *= C_rhs.C_ba_;
            this->reproject(false);
            return *this;
        }
        
        // -----------------------------------------------------------------------------
        // Right-hand multiplication with another rotation. 
        // -----------------------------------------------------------------------------

        Rotation Rotation::operator*(const Rotation& C_rhs) const {
            Rotation result(*this);
            result *= C_rhs;
            return result;
        }

        // -----------------------------------------------------------------------------
        // In-place right-hand multiplication with the inverse of another rotation. 
        // -----------------------------------------------------------------------------

        Rotation& Rotation::operator/=(const Rotation& C_rhs) {
            C_ba_ *= C_rhs.C_ba_.transpose();
            this->reproject(false);
            return *this;
        }

        // -----------------------------------------------------------------------------
        // Right-hand multiplication with the inverse of another rotation. 
        // -----------------------------------------------------------------------------

        Rotation Rotation::operator/(const Rotation& C_rhs) const {
            Rotation temp(*this);
            temp /= C_rhs;
            return temp;
        }

        // -----------------------------------------------------------------------------
        // Apply rotation to a 3D point vector
        // -----------------------------------------------------------------------------

        Eigen::Vector3d Rotation::operator*(const Eigen::Ref<const Eigen::Vector3d>& p_a) const {
            return C_ba_ * p_a;
        }

    }   // namespace se3
}   // namespace math

std::ostream& operator<<(std::ostream& out, const math::so3::Rotation& T) {
    out << "\n" << T.matrix() << "\n";
    return out;
}