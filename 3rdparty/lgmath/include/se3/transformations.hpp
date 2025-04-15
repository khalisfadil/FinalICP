#pragma once

#include <Eigen/Dense>

namespace math {
    namespace se3 {

        // Represents a rigid body transformation in SE(3).
        class Transformation {
        public:
            // Default constructor (identity transformation)
            Transformation();

            // Copy constructor
            Transformation(const Transformation&) = default;

            // Move constructor
            Transformation(Transformation&&) = default;

            // Construct transformation from a 4x4 homogeneous transformation matrix
            explicit Transformation(const Eigen::Matrix4d& T);

            // Construct transformation from rotation and translation components
            explicit Transformation(const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina);

            // Construct transformation from a 6D vector using the exponential map
            explicit Transformation(const Eigen::Matrix<double, 6, 1>& xi_ab, unsigned int numTerms = 0);

            // Construct transformation from a general Eigen vector
            explicit Transformation(const Eigen::VectorXd& xi_ab);

            // Destructor
            virtual ~Transformation() = default;

            // Copy assignment operator
            virtual Transformation& operator=(const Transformation&) = default;

            // Move assignment operator
            virtual Transformation& operator=(Transformation&& T) = default;

            // Gets basic matrix representation of the transformation
            Eigen::Matrix4d matrix() const;

            // Gets the underlying rotation matrix
            const Eigen::Matrix3d& C_ba() const;

            // Gets the translation vector r_ab_inb
            const Eigen::Vector3d& r_ab_inb() const;

            // Computes r_ba_ina = -C_ba.transpose() * r_ab_inb
            Eigen::Vector3d r_ba_ina() const;

            // Compute the logarithmic map (inverse of the exponential map)
            Eigen::Matrix<double, 6, 1> vec() const;

            // Compute the inverse of the transformation
            Transformation inverse() const;

            // Compute the 6x6 adjoint transformation matrix
            Eigen::Matrix<double, 6, 6> adjoint() const;

            // Reprojects the transformation matrix onto SE(3)
            void reproject(bool force = true);

            // Intrasound right-hand side multiplication with another transformation
            virtual Transformation& operator*=(const Transformation& T_rhs);

            // Right-hand side multiplication with another transformation
            virtual Transformation operator*(const Transformation& T_rhs) const;

            // In-place right-hand side multiplication with the inverse of another transformation
            virtual Transformation& operator/=(const Transformation& T_rhs);

            // Right-hand side multiplication with the inverse of another transformation
            virtual Transformation operator/(const Transformation& T_rhs) const;

            // Right-hand side multiplication with a homogeneous vector (4D point)
            Eigen::Vector4d operator*(const Eigen::Ref<const Eigen::Vector4d>& p_a) const;

        private:
            Eigen::Matrix3d C_ba_;      // Rotation matrix from frame a to frame b
            Eigen::Vector3d r_ab_inb_;  // Translation vector (position of frame a in frame b, expressed in frame b)
        };
    } // namespace se3
} // namespace math

// Print transformation
std::ostream& operator<<(std::ostream& out, const math::se3::Transformation& T);