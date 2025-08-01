#pragma once

#include <Eigen/Core>

#include <trajectory/constacc/priorfactor.hpp>
#include <trajectory/constacc/variable.hpp>
#include <trajectory/singer/helper.hpp>

namespace finalicp {
    namespace traj {
        namespace singer {

        class PriorFactor : public traj::const_acc::PriorFactor {
            public:
                using Ptr = std::shared_ptr<PriorFactor>;
                using ConstPtr = std::shared_ptr<const PriorFactor>;
                using Variable = traj::const_acc::Variable;

                // ###########################################################
                // MakeShared
                // ###########################################################

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad) {
                    return std::make_shared<PriorFactor>(knot1, knot2, ad);
                }
                
                // ###########################################################
                // PriorFactor
                // ###########################################################

                //Constructs an `AccelerationExtrapolator` instance.
                PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad)
                    : traj::const_acc::PriorFactor(knot1, knot2), alpha_diag_(ad) {
                    const double dt = (knot2_->time() - knot1_->time()).seconds();
                    Phi_ = getTran(dt, ad);
#ifdef DEBUG
                    // --- [IMPROVEMENT] Log factor creation and sanity check the transition matrix ---
                    std::cout << "[SINGER PriorFactor DEBUG] Creating Singer motion factor between knots at t="
                            << std::fixed << knot1_->time().seconds() << " and t=" << knot2_->time().seconds()
                            << " (dt=" << dt << "s)." << std::endl;

                    if (!Phi_.allFinite()) {
                        std::cerr << "[SINGER PriorFactor DEBUG] CRITICAL: Computed transition matrix Phi_ contains non-finite values!" << std::endl;
                    } else {
                        std::cout << "    - Transition matrix norm: " << Phi_.norm() << std::endl;
                    }
#endif
                }

            protected:
                const Eigen::Matrix<double, 6, 1> alpha_diag_;

                Eigen::Matrix<double, 18, 18> getJacKnot1_() const {
                    return getJacKnot1(knot1_, knot2_, alpha_diag_);
                }
            };
        }  // namespace singer
    }  // namespace traj
}  // namespace finalicp-