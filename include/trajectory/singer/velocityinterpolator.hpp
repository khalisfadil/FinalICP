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

                    // Q and Transition matrix
                    const auto Q_tau = getQ(tau, ad);
                    const auto Q_T = getQ(T, ad);
                    const auto Tran_kappa = getTran(kappa, ad);
                    const auto Tran_tau = getTran(tau, ad);
                    const auto Tran_T = getTran(T, ad);

                    // Calculate interpolation values
                    omega_ = Q_tau * Tran_kappa.transpose() * Q_T.inverse();
                    lambda_ = Tran_tau - omega_ * Tran_T;
                }
            };
        }  // namespace singer
    }  // namespace traj
}  // namespace finalicp-