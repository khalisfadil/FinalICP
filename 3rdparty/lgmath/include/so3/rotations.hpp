#pragma once

#include <Eigen/Dense>

namespace math{
    namespace so3{

            //lightweight rotation matrix class for efficient SO(3) operations.
            class Rotation {
                public:

                    //Default constructor (identity rotation).
                    Rotation();

                    //Copy constructor.
                    Rotation(const Rotation&) = default;

                    //Move constructor.
                    Rotation(Rotation&&) = default;

                    //Construct from an existing Eigen 3x3 matrix.
                    explicit Rotation(const Eigen::Matrix3d& C);

                    //Constructor. The rotation will be C_ba = vec2rot(aaxis_ab)
                    explicit Rotation(const Eigen::Vector3d& aaxis_ab, unsigned int numTerms = 0);

                    //Constructor. The rotation will be C_ba = vec2rot(aaxis_ab)
                    explicit Rotation(const Eigen::VectorXd& aaxis_ab);

                    //Destructor.
                    virtual ~Rotation() = default;

                    //Copy assignment operator.
                    virtual Rotation& operator=(const Rotation&) = default;

                    //Move assignment operator.
                    virtual Rotation& operator=(Rotation&&) = default;

                    //Gets the underlying rotation matrix 
                    const Eigen::Matrix3d& matrix() const;

                    //Get the corresponding Lie algebra using the logarithmic map
                    Eigen::Vector3d vec() const;

                    //Get the inverse (transpose) matrix
                    Rotation inverse() const;

                    //Reproject the rotation matrix back onto SO(3).
                    void reproject(bool force = true);

                    // In-place right-hand side multiply C_rhs
                    virtual Rotation& operator*=(const Rotation& C_rhs);

                    //Right-hand side multiply C_rhs
                    virtual Rotation operator*(const Rotation& C_rhs) const;

                    //In-place right-hand side multiply the inverse of C_rhs
                    virtual Rotation& operator/=(const Rotation& C_rhs);

                    //Right-hand side multiply the inverse of C_rhs
                    virtual Rotation operator/(const Rotation& C_rhs) const;

                    //Right-hand side multiply the point vector p_a
                    Eigen::Vector3d operator*(const Eigen::Ref<const Eigen::Vector3d>& p_a) const;

                private:
                    
                    Eigen::Matrix3d C_ba_;      //Rotation matrix from a to b
            };
    }   // namespace so3
}   // namespace math

//print transformation
std::ostream& operator<<(std::ostream& out, const math::so3::Rotation& T);