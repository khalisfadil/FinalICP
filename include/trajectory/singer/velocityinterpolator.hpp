#pragma once

#include <Eigen/Core>

#include <trajectory/constacc/variable.hpp>
#include <trajectory/constacc/velocityinterpolator.hpp>
#include <trajectory/singer/helper.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace singer {

        class VelocityInterpolator: public traj::const_acc::VelocityInterpolator {
            public:
                using Ptr = std::shared_ptr<VelocityInterpolator>;
                using ConstPtr = std::shared_ptr<const VelocityInterpolator>;
                using Variable = traj::const_acc::Variable;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot1,const Variable::ConstPtr& knot2,
                                        const Eigen::Matrix<double, 6, 1>& ad) {
                    return std::make_shared<VelocityInterpolator>(time, knot1, knot2, ad);
                }

                //Constructs an `AccelerationExtrapolator` instance.
                VelocityInterpolator(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad)
                    : traj::const_acc::VelocityInterpolator(time, knot1, knot2) {
                    // Calculate time constants
                    const double T = (knot2->time() - knot1->time()).seconds();
                    const double tau = (time - knot1->time()).seconds();
                    const double kappa = (knot2->time() - time).seconds();

#ifdef DEBUG
                    std::cout << " interpolating velocity with Singer model. Interval T: " << T << "s, at tau: " << tau << "s." << std::endl;
                    if (T <= 0) {
                        std::cerr << "[SINGER DEBUG | VelocityInterpolator] CRITICAL: Total time interval T is zero or negative!" << std::endl;
                    }
#endif

                    // Q and Transition matrix
                    const auto Q_tau = getQ(tau, ad);
                    const auto Q_T = getQ(T, ad);
                    const auto Tran_kappa = getTran(kappa, ad);
                    const auto Tran_tau = getTran(tau, ad);
                    const auto Tran_T = getTran(T, ad);

#ifdef DEBUG
                    // --- [IMPROVEMENT] Sanity-check matrix inversion ---
                    Eigen::FullPivLU<Eigen::Matrix<double, 18, 18>> lu(Q_T);
                    if (!lu.isInvertible()) {
                        std::cerr << "[SINGER DEBUG | VelocityInterpolator] CRITICAL: Process noise matrix Q_T is not invertible! Cannot compute interpolation matrices." << std::endl;
                        // Set to identity to avoid crashing, but the result will be wrong.
                        omega_.setIdentity();
                        lambda_.setIdentity();
                        return;
                    }
#endif

                    // Calculate interpolation values
                    omega_ = Q_tau * Tran_kappa.transpose() * Q_T.inverse();
                    lambda_ = Tran_tau - omega_ * Tran_T;
#ifdef DEBUG
                    // --- [IMPROVEMENT] Sanity-check final interpolation matrices ---
                    if (!omega_.allFinite() || !lambda_.allFinite()) {
                        std::cerr << "[SINGER DEBUG | VelocityInterpolator] CRITICAL: Final interpolation matrices (omega/lambda) contain non-finite values!" << std::endl;
                    } else {
                        std::cout << "    - Omega norm: " << omega_.norm() << ", Lambda norm: " << lambda_.norm() << std::endl;
                    }
#endif
                }
            };
        }  // namespace singer
    }  // namespace traj
}  // namespace finalicp-