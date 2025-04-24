#pragma once

#include <Eigen/Core>

#include <trajectory/constacc/posextrapolator.hpp>
#include <trajectory/constacc/variable.hpp>
#include <trajectory/singer/helper.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace singer {

        class PoseExtrapolator : public traj::const_acc::PoseExtrapolator {
            public:
                using Ptr = std::shared_ptr<PoseExtrapolator>;
                using ConstPtr = std::shared_ptr<const PoseExtrapolator>;
                using Variable = traj::const_acc::Variable;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot, const Eigen::Matrix<double, 6, 1>& ad) {
                    return std::make_shared<PoseExtrapolator>(time, knot, ad);
                }

                //Constructs an `AccelerationExtrapolator` instance.
                PoseExtrapolator(const Time time, const Variable::ConstPtr& knot,
                                const Eigen::Matrix<double, 6, 1>& ad)
                    : traj::const_acc::PoseExtrapolator(time, knot) {
                    const double tau = (time - knot->time()).seconds();
                    Phi_ = getTran(tau, ad);
                }
            };
        }  // namespace singer
    }  // namespace traj
}  // namespace finalicp-