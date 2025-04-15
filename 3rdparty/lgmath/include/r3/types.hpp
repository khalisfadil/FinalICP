#pragma once

#include <Eigen/Dense>

namespace math{
    namespace r3{
        using Point = Eigen::Vector3d;                                              //Represents a **3D point** in Euclidean space **ℝ³**.
        using PointRef = Eigen::Ref<Point>;                                         //Mutable reference to a **3D point**.
        using PointConstRef = Eigen::Ref<const Point>;                              //mmutable reference to a **3D point**.

        using HPoint = Eigen::Vector4d;                                             //Represents a **3D homogeneous point** (4D vector).
        using HPointRef = Eigen::Ref<HPoint>;                                       //Mutable reference to a **homogeneous point**.
        using HPointConstRef = Eigen::Ref<const HPoint>;                            //Immutable reference to a **homogeneous point**.

        using CovarianceMatrix = Eigen::Matrix3d;                                   //3D Point Covariance Matrix Representations**
        using CovarianceMatrixRef = Eigen::Ref<CovarianceMatrix>;                   //Mutable reference to a **3D covariance matrix**.
        using CovarianceMatrixConstRef = Eigen::Ref<const CovarianceMatrix>;        //Immutable reference to a **3D covariance matrix**.
    }   // namespace r3
}   // namespace math