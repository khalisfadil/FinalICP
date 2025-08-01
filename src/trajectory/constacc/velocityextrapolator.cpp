#include <trajectory/constacc/velocityextrapolator.hpp>
#include <trajectory/constacc/helper.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto VelocityExtrapolator::MakeShared(const Time time, const Variable::ConstPtr& knot) -> Ptr {
                return std::make_shared<VelocityExtrapolator>(time, knot);
            }

            // ###########################################################
            // VelocityExtrapolator
            // ###########################################################

            VelocityExtrapolator::VelocityExtrapolator(const Time time, const Variable::ConstPtr& knot)
                : knot_(knot) {
                const double tau = (time - knot->time()).seconds();
                Phi_ = getTran(tau);
#ifdef DEBUG
                // --- [IMPROVEMENT] Log creation and sanity-check the transition matrix ---
                std::cout << " extrapolating velocity with const_acc model over dt = " << tau << "s." << std::endl;
                if (tau < 0) {
                    std::cerr << "[CONSTACC VelocityExtrapolator DEBUG] WARNING: Extrapolating with a negative time delta (tau)!" << std::endl;
                }
                if (!Phi_.allFinite()) {
                    std::cerr << "[CONSTACC VelocityExtrapolator DEBUG] CRITICAL: Computed transition matrix Phi_ contains non-finite values!" << std::endl;
                }
#endif
            }

            // ###########################################################
            // active
            // ###########################################################

            bool VelocityExtrapolator::active() const {
                return knot_->velocity()->active() || knot_->acceleration()->active();
            }

            // ###########################################################
            // getRelatedVarKeys
            // ###########################################################

            void VelocityExtrapolator::getRelatedVarKeys(KeySet& keys) const {
                knot_->velocity()->getRelatedVarKeys(keys);
                knot_->acceleration()->getRelatedVarKeys(keys);
            }

            // ###########################################################
            // value
            // ###########################################################

            auto VelocityExtrapolator::value() const -> OutType {
                const auto knot_vel = knot_->velocity()->value();
                const auto knot_accel = knot_->acceleration()->value();
#ifdef DEBUG
                // --- [IMPROVEMENT] Check for numerical instability in the forward pass ---
                if (!knot_vel.allFinite() || !knot_accel.allFinite()) {
                    std::cerr << "[CONSTACC VelocityExtrapolator DEBUG | value] CRITICAL: Input knot state is non-finite!" << std::endl;
                }
#endif
                const Eigen::Matrix<double, 6, 1> xi_j1 = Phi_.block<6, 6>(6, 6) * knot_vel + Phi_.block<6, 6>(6, 12) * knot_accel;
#ifdef DEBUG
                if (!xi_j1.allFinite()) {
                    std::cerr << "[CONSTACC VelocityExtrapolator DEBUG | value] CRITICAL: Output extrapolated velocity is non-finite!" << std::endl;
                }
#endif
                return OutType(xi_j1);

                const Eigen::Matrix<double, 6, 1> xi_j1 = Phi_.block<6, 6>(6, 6) * knot_->velocity()->value() + Phi_.block<6, 6>(6, 12) * knot_->acceleration()->value();
                return OutType(xi_j1);  // approximation holds as long as xi_i1 is small.
            }

            // ###########################################################
            // forward
            // ###########################################################

            auto VelocityExtrapolator::forward() const -> Node<OutType>::Ptr {
                const auto w = knot_->velocity()->forward();
                const auto dw = knot_->acceleration()->forward();
                const Eigen::Matrix<double, 6, 1> xi_j1 = Phi_.block<6, 6>(6, 6) * w->value() + Phi_.block<6, 6>(6, 12) * dw->value();
                OutType w_i = xi_j1;  // approximation holds as long as xi_i1 is small.
                const auto node = Node<OutType>::MakeShared(w_i);
                node->addChild(w);
                node->addChild(dw);
                return node;
            }

            // ###########################################################
            // backward
            // ###########################################################

            void VelocityExtrapolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                if (!active()) return;
#ifdef DEBUG
                if (!lhs.allFinite()) {
                    std::cerr << "[CONSTACC VelocityExtrapolator DEBUG | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                }
#endif
                if (knot_->velocity()->active()) {
                    const auto w = std::static_pointer_cast<Node<InVelType>>(node->at(1));
                    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(6, 6);
#ifdef DEBUG
                if (!new_lhs.allFinite()) std::cerr << "[CONSTACC VelocityExtrapolator DEBUG | backward] CRITICAL: Derivative to velocity is non-finite!" << std::endl;
#endif
                    knot_->velocity()->backward(new_lhs, w, jacs);
                }
                if (knot_->acceleration()->active()) {
                    const auto dw = std::static_pointer_cast<Node<InAccType>>(node->at(1));
                    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(6, 12);
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC VelocityExtrapolator DEBUG | backward] CRITICAL: Derivative to acceleration is non-finite!" << std::endl;
#endif
                    knot_->acceleration()->backward(new_lhs, dw, jacs);
                }
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace finalicp