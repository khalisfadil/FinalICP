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

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad) {
                    return std::make_shared<PriorFactor>(knot1, knot2, ad);
                }

                //Constructs an `AccelerationExtrapolator` instance.
                PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad)
                    : traj::const_acc::PriorFactor(knot1, knot2), alpha_diag_(ad) {
                    const double dt = (knot2_->time() - knot1_->time()).seconds();
                    Phi_ = getTran(dt, ad);
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