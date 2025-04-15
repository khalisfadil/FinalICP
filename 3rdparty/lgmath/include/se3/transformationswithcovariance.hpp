#pragma once

#include <Eigen/Core>

#include <se3/transformations.hpp>

namespace math{
    namespace se3{

        //A transformation matrix class with associated covariance.
        class TransformationWithCovariance : public Transformation {
            public:

                //Default constructor (optionally initializes covariance to zero)
                TransformationWithCovariance(bool initCovarianceToZero = false);

                //Copy constructor
                TransformationWithCovariance(const TransformationWithCovariance&) = default;

                //Move constructor
                TransformationWithCovariance(TransformationWithCovariance&&) = default;

                //Copy constructor from basic Transformation 
                TransformationWithCovariance(const Transformation& T, bool initCovarianceToZero = false);

                //Move constructor from basic Transformation
                TransformationWithCovariance(Transformation&& T, bool initCovarianceToZero = false);

                //Copy constructor from basic Transformation with covariance
                TransformationWithCovariance(const Transformation& T, const Eigen::Matrix<double, 6, 6>& covariance);

                //Constructor from a 4x4 transformation matrix
                TransformationWithCovariance(const Eigen::Matrix4d& T);

                //Constructor from a 4x4 transformation matrix with covariance
                TransformationWithCovariance(const Eigen::Matrix4d& T, const Eigen::Matrix<double, 6, 6>& covariance);

                //Constructor.
                TransformationWithCovariance(const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina);

                //Constructor with covariance.
                TransformationWithCovariance(const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina, const Eigen::Matrix<double, 6, 6>& covariance);

                //Constructor.
                TransformationWithCovariance(const Eigen::Matrix<double, 6, 1>& xi_ab, unsigned int numTerms = 0);

                //Constructor with covariance.
                TransformationWithCovariance(const Eigen::Matrix<double, 6, 1>& xi_ab, const Eigen::Matrix<double, 6, 6>& covariance, unsigned int numTerms = 0);  

                //Constructor.
                TransformationWithCovariance(const Eigen::VectorXd& xi_ab);

                //Constructor.
                TransformationWithCovariance(const Eigen::VectorXd& xi_ab, const Eigen::Matrix<double, 6, 6>& covariance); 

                //Destructor. Default implementation.
                ~TransformationWithCovariance() override = default;

                //Copy assignment operator.
                TransformationWithCovariance& operator=(const TransformationWithCovariance&) = default;

                //Move assignment operator.
                TransformationWithCovariance& operator=(TransformationWithCovariance&& T) = default;

                //Move assignment operator.
                TransformationWithCovariance& operator=(const Transformation& T) noexcept override;

                //Move assignment operator.
                TransformationWithCovariance& operator=(Transformation&& T) noexcept override;

                //Gets the underlying covariance matrix
                const Eigen::Matrix<double, 6, 6>& cov() const;

                //Returns whether or not a covariance has been set.
                bool covarianceSet() const;

                //Sets the underlying covariance matrix 
                void setCovariance(const Eigen::Matrix<double, 6, 6>& covariance);

                //Sets the underlying covariance matrix 
                void setZeroCovariance();

                //Sets the underlying covariance matrix 
                TransformationWithCovariance inverse() const;

                //In-place right-hand side multiply T_rhs.
                TransformationWithCovariance& operator*=(const TransformationWithCovariance& T_rhs);

                //In-place right-hand side multiply basic (certain) T_rhs
                TransformationWithCovariance& operator*=(const Transformation& T_rhs) override;

                //In-place right-hand side multiply the inverse of T_rhs
                TransformationWithCovariance& operator/=(const TransformationWithCovariance& T_rhs);

                //In-place right-hand side multiply the inverse of a basic (certain)
                TransformationWithCovariance& operator/=(const Transformation& T_rhs) override;

            private:
                //Covariance
                Eigen::Matrix<double, 6, 6> covariance_;

                //Covariance flag
                bool covarianceSet_;
        };
        
        //Multiplication of two TransformWithCovariance
        TransformationWithCovariance operator*(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs);

        //brief Multiplication of TransformWithCovariance by Transform
        TransformationWithCovariance operator*(TransformationWithCovariance T_lhs, const Transformation& T_rhs);

        //brief Multiplication of Transform by TransformWithCovariance
        TransformationWithCovariance operator*(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);

        //brief Multiplication of TransformWithCovariance by inverse
        TransformationWithCovariance operator/(TransformationWithCovariance T_lhs, const TransformationWithCovariance& T_rhs);

        //brief Multiplication of TransformWithCovariance by inverse Transform
        TransformationWithCovariance operator/(TransformationWithCovariance T_lhs, const Transformation& T_rhs);

        //brief Multiplication of Transform by inverse TransformWithCovariance
        TransformationWithCovariance operator/(const Transformation& T_lhs, const TransformationWithCovariance& T_rhs);


    }   // namespace se3
}   // namespace math 

//brief print transformation
std::ostream& operator<<(std::ostream& out,const math::se3::TransformationWithCovariance& T);
