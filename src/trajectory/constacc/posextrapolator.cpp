#include <trajectory/constacc/posextrapolator.hpp>
#include <trajectory/constacc/helper.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

            // ###########################################################
            // MakeShared
            // ###########################################################

            PoseExtrapolator::Ptr PoseExtrapolator::MakeShared(const Time time, const Variable::ConstPtr& knot) {
                return std::make_shared<PoseExtrapolator>(time, knot);
            }

            // ###########################################################
            // PoseExtrapolator
            // ###########################################################

            PoseExtrapolator::PoseExtrapolator(const Time time, const Variable::ConstPtr& knot)
                : knot_(knot) {
                const double tau = (time - knot->time()).seconds();
                Phi_ = getTran(tau);
#ifdef DEBUG
                // --- [IMPROVEMENT] Log creation and sanity-check the transition matrix ---
                std::cout << " extrapolating pose with const_acc model over dt = " << tau << "s." << std::endl;
                if (tau < 0) {
                    std::cerr << "[CONSTACC PoseExtrapolator DEBUG] WARNING: Extrapolating with a negative time delta (tau)!" << std::endl;
                }
                if (!Phi_.allFinite()) {
                    std::cerr << "[CONSTACC PoseExtrapolator DEBUG] CRITICAL: Computed transition matrix Phi_ contains non-finite values!" << std::endl;
                }
#endif
            }

            // ###########################################################
            // active
            // ###########################################################

            bool PoseExtrapolator::active() const {
                return knot_->pose()->active() || knot_->velocity()->active() || knot_->acceleration()->active();
            }

            // ###########################################################
            // getRelatedVarKeys
            // ###########################################################

            void PoseExtrapolator::getRelatedVarKeys(KeySet& keys) const {
                knot_->pose()->getRelatedVarKeys(keys);
                knot_->velocity()->getRelatedVarKeys(keys);
                knot_->acceleration()->getRelatedVarKeys(keys);
            }

            // ###########################################################
            // value
            // ###########################################################

            auto PoseExtrapolator::value() const -> OutType {
                const auto knot_pose = knot_->pose()->value();
                const auto knot_vel = knot_->velocity()->value();
                const auto knot_accel = knot_->acceleration()->value();
#ifdef DEBUG
                if (!knot_pose.matrix().allFinite() || !knot_vel.allFinite() || !knot_accel.allFinite()) {
                    std::cerr << "[CONSTACC PoseExtrapolator DEBUG | value] CRITICAL: Input knot state is non-finite!" << std::endl;
                }
#endif
                const math::se3::Transformation T_i1(Eigen::Matrix<double, 6, 1>(Phi_.block<6, 6>(0, 6) * knot_vel + Phi_.block<6, 6>(0, 12) * knot_accel));
                OutType T_i0 = T_i1 * knot_pose;
#ifdef DEBUG
                if (!T_i0.matrix().allFinite()) {
                    std::cerr << "[CONSTACC PoseExtrapolator DEBUG | value] CRITICAL: Output extrapolated pose is non-finite!" << std::endl;
                }
#endif
                return T_i0;
            }

            // ###########################################################
            // forward
            // ###########################################################

            auto PoseExtrapolator::forward() const -> Node<OutType>::Ptr {
                const auto T = knot_->pose()->forward();
                const auto w = knot_->velocity()->forward();
                const auto dw = knot_->acceleration()->forward();
                const math::se3::Transformation T_i1(Eigen::Matrix<double, 6, 1>(Phi_.block<6, 6>(0, 6) * w->value() + Phi_.block<6, 6>(0, 12) * dw->value()));
                OutType T_i0 = T_i1 * T->value();
                const auto node = Node<OutType>::MakeShared(T_i0);
                node->addChild(T);
                node->addChild(w);
                node->addChild(dw);
                return node;
            }

            // ###########################################################
            // backward
            // ###########################################################

            void PoseExtrapolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                if (!active()) return;
#ifdef DEBUG
                if (!lhs.allFinite()) {
                    std::cerr << "[CONSTACC PoseExtrapolator DEBUG | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                }
#endif
                const auto w = knot_->velocity()->value();
                const auto dw = knot_->acceleration()->value();
                const Eigen::Matrix<double, 6, 1> xi_i1 = Phi_.block<6, 6>(0, 6) * w + Phi_.block<6, 6>(0, 12) * dw;
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                const math::se3::Transformation T_i1(xi_i1);
                if (knot_->pose()->active()) {
                    const auto T_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
                    Eigen::MatrixXd new_lhs = lhs * T_i1.adjoint();
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseExtrapolator DEBUG | backward] CRITICAL: Derivative to T is non-finite!" << std::endl;
#endif
                    knot_->pose()->backward(new_lhs, T_, jacs);
                }
                if (knot_->velocity()->active()) {
                    const auto w = std::static_pointer_cast<Node<InVelType>>(node->at(1));
                    Eigen::MatrixXd new_lhs = lhs * J_i1 * Phi_.block<6, 6>(0, 6);
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseExtrapolator DEBUG | backward] CRITICAL: Derivative to w is non-finite!" << std::endl;
#endif
                    knot_->velocity()->backward(new_lhs, w, jacs);
                }
                if (knot_->acceleration()->active()) {
                    const auto dw = std::static_pointer_cast<Node<InAccType>>(node->at(2));
                    Eigen::MatrixXd new_lhs = lhs * J_i1 * Phi_.block<6, 6>(0, 12);
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseExtrapolator DEBUG | backward] CRITICAL: Derivative to dw is non-finite!" << std::endl;
#endif
                    knot_->acceleration()->backward(new_lhs, dw, jacs);
                }
            }
        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp