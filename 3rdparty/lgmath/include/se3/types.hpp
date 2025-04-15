#pragma once

#include <Eigen/Core>

namespace math{
    namespace se3{
            using TranslationVector = Eigen::Vector3d;                      //A 3D translation vector.
            using LieAlgebra = Eigen::Matrix<double, 6, 1>;                 //A Lie algebra vector representing an element of se(3).
            using LieAlgebraCovariance = Eigen::Matrix<double, 6, 6>;       //Covariance matrix associated with a Lie algebra vector.
            using TransformationMatrix = Eigen::Matrix4d;                   //A 4x4 transformation matrix representing an SE(3) element.
    }   // namespace se3
}   // namespace math