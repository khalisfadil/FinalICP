#pragma once

#include <Eigen/Core>

namespace math{
    namespace so3{
            using AxisAngle = Eigen::Vector3d;                      //Represents a 3D axis-angle rotation vector.
            using RotationMatrix = Eigen::Matrix3d;                 //Represents a 3x3 rotation matrix.
    }   // namespace so3
}   // namespace math