#pragma once

#include <Eigen/Core>


namespace math {

    /// Mathematical Constants (High Precision)
    namespace constants {
        constexpr double TWO_PI = 6.283185307179586;         ///< 2π
        constexpr double PI = 3.141592653589793;             ///< π
        constexpr double PI_DIV_TWO = 1.570796326794897;     ///< π/2
        constexpr double PI_DIV_FOUR = 0.785398163397448;    ///< π/4
        constexpr double ONE_DIV_PI = 0.318309886183791;     ///< 1/π
        constexpr double ONE_DIV_TWO_PI = 0.159154943091895; ///< 1/(2π)
        constexpr double DEG2RAD = 0.017453292519943;        ///< π/180 (degrees → radians)
        constexpr double RAD2DEG = 57.295779513082323;       ///< 180/π (radians → degrees)
    }  // namespace constants

    /// Common Math Functions
    namespace common {

        /**
         * @brief Wraps an angle into the range [-π, π].
         * @param radians Angle in radians.
         * @return Wrapped angle in [-π, π].
         */
        double angleMod(double radians);

        /**
         * @brief Converts degrees to radians.
         * @param degrees Angle in degrees.
         * @return Angle in radians.
         */
        double deg2rad(double degrees);

        /**
         * @brief Converts radians to degrees.
         * @param radians Angle in radians.
         * @return Angle in degrees.
         */
        double rad2deg(double radians);

        /**
         * @brief Compares two floating-point values for near equality.
         * @param a First value.
         * @param b Second value.
         * @param tol Tolerance for comparison (default: 1e-6).
         * @return True if |a - b| ≤ tol, false otherwise.
         */
        bool nearEqual(double a, double b, double tol = 1e-6);

        /**
         * @brief Compares two Eigen matrices for near equality.
         * @param A First matrix.
         * @param B Second matrix.
         * @param tol Tolerance for comparison (default: 1e-6).
         * @return True if matrices have same size and are approximately equal, false otherwise.
         */
        bool nearEqual(const Eigen::MatrixXd& A,
                    const Eigen::MatrixXd& B,
                    double tol = 1e-6);

        /**
         * @brief Compares two angles (in radians) for near equality, accounting for wrapping.
         * @param radA First angle in radians.
         * @param radB Second angle in radians.
         * @param tol Tolerance for comparison (default: 1e-6).
         * @return True if angles are approximately equal within [-π, π], false otherwise.
         */
        bool nearEqualAngle(double radA, double radB, double tol = 1e-6);

        /**
         * @brief Compares two axis-angle vectors for near equality.
         * @param aaxis1 First axis-angle vector.
         * @param aaxis2 Second axis-angle vector.
         * @param tol Tolerance for comparison (default: 1e-6).
         * @return True if axes and angles are approximately equal, false otherwise.
         */
        bool nearEqualAxisAngle(const Eigen::Vector3d& aaxis1,
                                const Eigen::Vector3d& aaxis2,
                                double tol = 1e-6);

        /**
         * @brief Compares two SE(3) Lie algebra vectors for near equality.
         * @param vec1 First Lie algebra vector (6x1: translation + rotation).
         * @param vec2 Second Lie algebra vector (6x1: translation + rotation).
         * @param tol Tolerance for comparison (default: 1e-6).
         * @return True if translation and rotation components are approximately equal, false otherwise.
         */
        bool nearEqualLieAlg(const Eigen::Matrix<double, 6, 1>& vec1,
                            const Eigen::Matrix<double, 6, 1>& vec2,
                            double tol = 1e-6);

    }  // namespace common
}  // namespace math