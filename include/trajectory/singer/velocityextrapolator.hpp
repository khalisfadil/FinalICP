#pragma once

#include <Eigen/Core>

#include <trajectory/constacc/variable.hpp>
#include <trajectory/constacc/velocityextrapolator.hpp>
#include <trajectory/singer/helper.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace singer {

        class VelocityExtrapolator: public traj::const_acc::VelocityExtrapolator {
            public:
                using Ptr = std::shared_ptr<VelocityExtrapolator>;
                using ConstPtr = std::shared_ptr<const VelocityExtrapolator>;
                using Variable = traj::const_acc::Variable;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot, const Eigen::Matrix<double, 6, 1>& ad) {
                    return std::make_shared<VelocityExtrapolator>(time, knot, ad);
                }

                //Constructs an `AccelerationExtrapolator` instance.
                VelocityExtrapolator(const Time time, const Variable::ConstPtr& knot, const Eigen::Matrix<double, 6, 1>& ad)
                    : traj::const_acc::VelocityExtrapolator(time, knot) {
                    const double tau = (time - knot->time()).seconds();
                    Phi_ = getTran(tau, ad);
#ifdef DEBUG
                    // --- [IMPROVEMENT] Log creation and sanity-check the transition matrix ---
                    std::cout << " extrapolating velocity with Singer model over dt = " << tau << "s." << std::endl;
                    if (!Phi_.allFinite()) {
                        std::cerr << "[SINGER DEBUG | VelocityExtrapolator] CRITICAL: Computed transition matrix Phi_ contains non-finite values!" << std::endl;
                    } else {
                        // Logging the norm is a good quick check for stability.
                        std::cout << "    - Transition matrix norm: " << Phi_.norm() << std::endl;
                    }
#endif
                }
            };
        }  // namespace singer
    }  // namespace traj
}  // namespace finalicp-