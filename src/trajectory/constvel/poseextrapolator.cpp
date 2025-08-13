#include <trajectory/constvel/poseextrapolator.hpp>
#include <trajectory/constvel/helper.hpp>

namespace finalicp {
    namespace traj {
        namespace const_vel {

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto PoseExtrapolator::MakeShared(const Time time, const Variable::ConstPtr& knot) -> Ptr {
                return std::make_shared<PoseExtrapolator>(time, knot);
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            PoseExtrapolator::PoseExtrapolator(const Time time, const Variable::ConstPtr& knot)
                : knot_(knot) {
                const double tau = (time - knot->time()).seconds();
                Phi_ = getTran(tau);
#ifdef DEBUG
                // --- [IMPROVEMENT] Log creation and sanity-check the transition matrix ---
                std::cout << "[CONSTVEL DEBUG | PoseExtrapolator] extrapolating pose with const_acc model over dt = " << tau << "s." << std::endl;
                if (tau < 0) {
                    std::cerr << "[CONSTVEL DEBUG | PoseExtrapolator] WARNING: Extrapolating with a negative time delta (tau)!" << std::endl;
                }
                if (!Phi_.allFinite()) {
                    std::cerr << "[CONSTVEL DEBUG | PoseExtrapolator] CRITICAL: Computed transition matrix Phi_ contains non-finite values!" << std::endl;
                }
#endif
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            bool PoseExtrapolator::active() const {
                return knot_->pose()->active() || knot_->velocity()->active();
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            void PoseExtrapolator::getRelatedVarKeys(KeySet& keys) const {
                knot_->pose()->getRelatedVarKeys(keys);
                knot_->velocity()->getRelatedVarKeys(keys);
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto PoseExtrapolator::value() const -> OutType {
                const math::se3::Transformation T_i1(Eigen::Matrix<double, 6, 1>(
                Phi_.block<6, 6>(0, 6) * knot_->velocity()->value()));
                return OutType(T_i1 * knot_->pose()->value());
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto PoseExtrapolator::forward() const -> Node<OutType>::Ptr {
                const auto T = knot_->pose()->forward();
                const auto w = knot_->velocity()->forward();
                const math::se3::Transformation T_i1(Eigen::Matrix<double, 6, 1>(Phi_.block<6, 6>(0, 6) * w->value()));
                OutType T_i0 = T_i1 * T->value();
                const auto node = Node<OutType>::MakeShared(T_i0);
                node->addChild(T);
                node->addChild(w);
                return node;
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            void PoseExtrapolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                if (!active()) return;
#ifdef DEBUG
                if (!lhs.allFinite()) {
                    std::cerr << "[CONSTVEL DEBUG | PoseExtrapolator | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                }
#endif
                const auto w = knot_->velocity()->value();
                const Eigen::Matrix<double, 6, 1> xi_i1 = Phi_.block<6, 6>(0, 6) * w;
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                const math::se3::Transformation T_i1(xi_i1);
                if (knot_->pose()->active()) {
                    const auto T_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
                    Eigen::MatrixXd new_lhs = lhs * T_i1.adjoint();
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTVEL DEBUG | PoseExtrapolator | backward] CRITICAL: Derivative to T is non-finite!" << std::endl;
#endif
                    knot_->pose()->backward(new_lhs, T_, jacs);
                }
                if (knot_->velocity()->active()) {
                    const auto w_ = std::static_pointer_cast<Node<InVelType>>(node->at(1));
                    Eigen::MatrixXd new_lhs = lhs * J_i1 * Phi_.block<6, 6>(0, 6);
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTVEL DEBUG | PoseExtrapolator | backward] CRITICAL: Derivative to dw is non-finite!" << std::endl;
#endif
                    knot_->velocity()->backward(new_lhs, w_, jacs);
                }
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace finalicp