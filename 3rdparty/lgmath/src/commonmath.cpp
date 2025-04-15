#include <commonmath.hpp>

#include <cmath>


namespace math {
    namespace common {

        // ----------------------------------------------------------------------------
        // Angle Wrapping
        // ----------------------------------------------------------------------------

        double angleMod(double radians) {
            return radians - constants::TWO_PI * std::round(radians * constants::ONE_DIV_TWO_PI);
        }

        // ----------------------------------------------------------------------------
        // Degree-Radian Conversions
        // ----------------------------------------------------------------------------

        double deg2rad(double degrees) {
            return degrees * constants::DEG2RAD;
        }

        double rad2deg(double radians) {
            return radians * constants::RAD2DEG;
        }

        // ----------------------------------------------------------------------------
        // Near Equality Comparisons
        // ----------------------------------------------------------------------------

        // Compare two doubles within tolerance
        bool nearEqual(double a, double b, double tol) {
            return std::fabs(a - b) <= tol;
        }

        // Compare two matrices within tolerance
        bool nearEqual(const Eigen::MatrixXd& A,
                    const Eigen::MatrixXd& B,
                    double tol) {
            return A.rows() == B.rows() && A.cols() == B.cols() && A.isApprox(B, tol);
        }

        // Compare two angles with wrapping
        bool nearEqualAngle(double radA, double radB, double tol) {
            return nearEqual(angleMod(radA - radB), 0.0, tol);
        }

        // Compare two axis-angle vectors
        bool nearEqualAxisAngle(const Eigen::Vector3d& aaxis1,
                                const Eigen::Vector3d& aaxis2,
                                double tol) {
            constexpr double EPSILON = 1e-12;

            double a1 = aaxis1.norm();
            double a2 = aaxis2.norm();

            if (a1 < EPSILON && a2 < EPSILON) {
                return true;
            }

            Eigen::Vector3d axis1 = aaxis1 / a1;
            Eigen::Vector3d axis2 = aaxis2 / a2;

            bool axes_equal = axis1.isApprox(axis2, tol) && nearEqualAngle(a1, a2, tol);
            bool axes_opposite = axis1.isApprox(-axis2, tol) && nearEqualAngle(a1, -a2, tol);

            return axes_equal || axes_opposite;
        }

        // Compare two SE(3) Lie algebra vectors
        bool nearEqualLieAlg(const Eigen::Matrix<double, 6, 1>& vec1,
                            const Eigen::Matrix<double, 6, 1>& vec2,
                            double tol) {
            return nearEqualAxisAngle(vec1.tail<3>(), vec2.tail<3>(), tol) &&
                nearEqual(vec1.head<3>(), vec2.head<3>(), tol);
        }

    }  // namespace common
}  // namespace liemath
