#pragma once

#include <Eigen/Core>

#include <trajectory/constacc/poseinterpolator.hpp>
#include <trajectory/constacc/variable.hpp>
#include <trajectory/singer/helper.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace singer {

        class PoseInterpolator : public traj::const_acc::PoseInterpolator {
            public:
                using Ptr = std::shared_ptr<PoseInterpolator>;
                using ConstPtr = std::shared_ptr<const PoseInterpolator>;
                using Variable = traj::const_acc::Variable;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad) {
                    return std::make_shared<PoseInterpolator>(time, knot1, knot2, ad);
                }

                //Constructs an `AccelerationExtrapolator` instance.
                PoseInterpolator(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad)
                    : traj::const_acc::PoseInterpolator(time, knot1, knot2) {
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