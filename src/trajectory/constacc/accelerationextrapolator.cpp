#include <trajectory/constacc/accelerationextrapolator.hpp>
#include <trajectory/constacc/helper.hpp>


namespace finalicp {
    namespace traj {
        namespace const_acc {

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto AccelerationExtrapolator::MakeShared(const Time time, const Variable::ConstPtr& knot)
                -> Ptr {
                return std::make_shared<AccelerationExtrapolator>(time, knot);
            }

            // ###########################################################
            // AccelerationExtrapolator
            // ###########################################################

            AccelerationExtrapolator::AccelerationExtrapolator(const Time time, const Variable::ConstPtr& knot)
                    : knot_(knot) {
                const double tau = (time - knot->time()).seconds();
                Phi_ = getTran(tau);
#ifdef DEBUG
                // --- [IMPROVEMENT] Log creation and sanity-check the transition matrix ---
                std::cout << " extrapolating acceleration with const_acc model over dt = " << tau << "s." << std::endl;
                if (tau < 0) {
                    std::cerr << "[CONSTACC AccelerationExtrapolator DEBUG] WARNING: Extrapolating with a negative time delta (tau)!" << std::endl;
                }
                if (!Phi_.allFinite()) {
                    std::cerr << "[CONSTACC AccelerationExtrapolator DEBUG] CRITICAL: Computed transition matrix Phi_ contains non-finite values!" << std::endl;
                }
#endif
            }

            // ###########################################################
            // active
            // ###########################################################

            bool AccelerationExtrapolator::active() const {
                return knot_->acceleration()->active();
            }

            // ###########################################################
            // getRelatedVarKeys
            // ###########################################################

            void AccelerationExtrapolator::getRelatedVarKeys(KeySet& keys) const {
                knot_->acceleration()->getRelatedVarKeys(keys);
            }

            // ###########################################################
            // value
            // ###########################################################

            auto AccelerationExtrapolator::value() const -> OutType {
                const auto input_accel = knot_->acceleration()->value();
                const Eigen::Matrix<double, 6, 1> xi_k1 = Phi_.block<6, 6>(12, 12) * input_accel;
#ifdef DEBUG
                // --- [IMPROVEMENT] Check for numerical instability in the forward pass ---
                if (!input_accel.allFinite()) {
                    std::cerr << "[CONSTACC AccelerationExtrapolator DEBUG | value] CRITICAL: Input acceleration is non-finite!" << std::endl;
                }
                if (!xi_k1.allFinite()) {
                    std::cerr << "[CONSTACC AccelerationExtrapolator DEBUG | value] CRITICAL: Output extrapolated acceleration is non-finite!" << std::endl;
                }
#endif
                return OutType(xi_k1);  // approximation holds as long as xi_i1 is small.
            }

            // ###########################################################
            // forward
            // ###########################################################

            auto AccelerationExtrapolator::forward() const -> Node<OutType>::Ptr {
                const auto dw = knot_->acceleration()->forward();
                const Eigen::Matrix<double, 6, 1> xi_k1 = Phi_.block<6, 6>(12, 12) * dw->value();
                OutType dw_i = xi_k1;  // approximation holds as long as xi_i1 is small.
                const auto node = Node<OutType>::MakeShared(dw_i);
                node->addChild(dw);
                return node;
            }

            // ###########################################################
            // backward
            // ###########################################################

            void AccelerationExtrapolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                if (!active()) return;
                if (knot_->acceleration()->active()) {
                    const auto dw = std::static_pointer_cast<Node<InAccType>>(node->at(0));
                    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(12, 12);
#ifdef DEBUG
                    // --- [IMPROVEMENT] Check derivatives in the backward pass ---
                    if (!lhs.allFinite()) {
                        std::cerr << "[CONSTACC AccelerationExtrapolator DEBUG | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                    }
                    if (!new_lhs.allFinite()) {
                        std::cerr << "[CONSTACC AccelerationExtrapolator DEBUG | backward] CRITICAL: Derivative propagated to child is non-finite!" << std::endl;
                    }
#endif
                    knot_->acceleration()->backward(new_lhs, dw, jacs);
                }
            }
        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp