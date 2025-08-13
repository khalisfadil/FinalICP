#include <trajectory/constvel/poseinterpolator.hpp>

#include <evaluable/se3/evaluables.hpp>
#include <evaluable/vspace/evaluables.hpp>
#include <trajectory/constvel/evaluable/jinvvelocityevaluator.hpp>
#include <trajectory/constvel/helper.hpp>

namespace finalicp {
    namespace traj {
        namespace const_vel {

            // ###########################################################
            // MakeShared
            // ###########################################################

            PoseInterpolator::Ptr PoseInterpolator::MakeShared(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
                return std::make_shared<PoseInterpolator>(time, knot1, knot2);
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            PoseInterpolator::PoseInterpolator(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                // Calculate time constants
                const double tau = (time - knot1->time()).seconds();
                const double T = (knot2->time() - knot1->time()).seconds();
#ifdef DEBUG
                // std::cout << " interpolating pose with const_acc model. Interval T: " << T << "s, at tau: " << tau << "s." << std::endl;
                if (T <= 0) {
                    std::cerr << "[CONSTVEL DEBUG | PoseInterpolator] CRITICAL: Total time interval T is zero or negative!" << std::endl;
                }
#endif
                const double ratio = tau / T;
                const double ratio2 = ratio * ratio;
                const double ratio3 = ratio2 * ratio;
                // Calculate 'psi' interpolation values
                psi11_ = 3.0 * ratio2 - 2.0 * ratio3;
                psi12_ = tau * (ratio2 - ratio);
                psi21_ = 6.0 * (ratio - ratio2) / T;
                psi22_ = 3.0 * ratio2 - 2.0 * ratio;
                // Calculate 'lambda' interpolation values
                lambda11_ = 1.0 - psi11_;
                lambda12_ = tau - T * psi11_ - psi12_;
                lambda21_ = -psi21_;
                lambda22_ = 1.0 - T * psi21_ - psi22_;
#ifdef DEBUG
                if (!omega_.allFinite()) {
                    std::cerr << "[CONSTVEL DEBUG | PoseInterpolator] CRITICAL: Final interpolation matrices lambda contain non-finite values!" << std::endl;
                }
#endif
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            bool PoseInterpolator::active() const {
                return knot1_->pose()->active() || knot1_->velocity()->active() || knot2_->pose()->active() || knot2_->velocity()->active();
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
                knot1_->pose()->getRelatedVarKeys(keys);
                knot1_->velocity()->getRelatedVarKeys(keys);
                knot2_->pose()->getRelatedVarKeys(keys);
                knot2_->velocity()->getRelatedVarKeys(keys);
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto PoseInterpolator::value() const -> OutType {
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
#ifdef DEBUG
                if (!w1.allFinite() || !w2.allFinite()) {
                    std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | value] CRITICAL: Input knot states are non-finite!" << std::endl;
                }
#endif
                // Get se3 algebra of relative matrix
                const auto xi_21 = (T2 / T1).vec();
                // Calculate the 6x6 associated Jacobian
                const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
                // Calculate interpolated relative se3 algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_21_inv * w2;
#ifdef DEBUG
                if (!xi_i1.allFinite()) {
                    std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | value] CRITICAL: Intermediate xi_i1 vector is non-finite!" << std::endl;
                }
#endif
                // Calculate interpolated relative transformation matrix
                const math::se3::Transformation T_i1(xi_i1);
                OutType T_i0 = T_i1 * T1;
#ifdef DEBUG
                if (!T_i0.matrix().allFinite()) {
                    std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | value] CRITICAL: Final interpolated pose is non-finite!" << std::endl;
                }
#endif
                return T_i0;
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
                const auto T1 = knot1_->pose()->forward();
                const auto w1 = knot1_->velocity()->forward();
                const auto T2 = knot2_->pose()->forward();
                const auto w2 = knot2_->velocity()->forward();
                // Get se3 algebra of relative matrix
                const auto xi_21 = (T2->value() / T1->value()).vec();
                // Calculate the 6x6 associated Jacobian
                const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
                // Calculate interpolated relative se3 algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * w1->value() + psi11_ * xi_21 + psi12_ * J_21_inv * w2->value();
                // Calculate interpolated relative transformation matrix
                const math::se3::Transformation T_i1(xi_i1);
                OutType T_i0 = T_i1 * T1->value();
                const auto node = Node<OutType>::MakeShared(T_i0);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(T2);
                node->addChild(w2);
                return node;
            }

            // ###########################################################
            // MakeShared
            // ###########################################################

            void PoseInterpolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                if (!active()) return;
#ifdef DEBUG
                if (!lhs.allFinite()) {
                    std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                }
#endif
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                // Get se3 algebra of relative matrix
                const auto xi_21 = (T2 / T1).vec();
                // Calculate the 6x6 associated Jacobian
                const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
                // Calculate interpolated relative se3 algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_21_inv * w2;
                // Calculate interpolated relative transformation matrix
                const math::se3::Transformation T_21(xi_21);
                const math::se3::Transformation T_i1(xi_i1);
                // Calculate the 6x6 Jacobian associated with the interpolated relative
                // transformation matrix
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);

                if (knot1_->pose()->active() || knot2_->pose()->active()) {
                    // Precompute part of jacobian matrices
                    const Eigen::Matrix<double, 6, 6> w = J_i1 * (psi11_ * J_21_inv + psi12_ * 0.5 * math::se3::curlyhat(w2) * J_21_inv);
                    if (knot1_->pose()->active()) {
                        const auto T1_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
                        Eigen::MatrixXd new_lhs = lhs * (-w * T_21.adjoint() + T_i1.adjoint());
#ifdef DEBUG
                        if (!new_lhs.allFinite()) std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | backward] CRITICAL: Derivative to T1 is non-finite!" << std::endl;
#endif
                        knot1_->pose()->backward(new_lhs, T1_, jacs);
                    }
                    if (knot2_->pose()->active()) {
                        const auto T2_ = std::static_pointer_cast<Node<InPoseType>>(node->at(2));
                        Eigen::MatrixXd new_lhs = lhs * w;
#ifdef DEBUG
                        if (!new_lhs.allFinite()) std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | backward] CRITICAL: Derivative to T1 is non-finite!" << std::endl;
#endif
                        knot2_->pose()->backward(new_lhs, T2_, jacs);
                    }
                }
                if (knot1_->velocity()->active()) {
                    const auto w1_ = std::static_pointer_cast<Node<InVelType>>(node->at(1));
                    Eigen::MatrixXd new_lhs = lhs * lambda12_ * J_i1;
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | backward] CRITICAL: Derivative to T1 is non-finite!" << std::endl;
#endif
                    knot1_->velocity()->backward(new_lhs, w1_, jacs);
                }
                if (knot2_->velocity()->active()) {
                    const auto w2_ = std::static_pointer_cast<Node<InVelType>>(node->at(3));
                    Eigen::MatrixXd new_lhs = lhs * psi12_ * J_i1 * J_21_inv;
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTVEL DEBUG | PoseInterpolator | backward] CRITICAL: Derivative to T1 is non-finite!" << std::endl;
#endif
                    knot2_->velocity()->backward(new_lhs, w2_, jacs);
                }
            }
        }  // namespace const_vel
    }  // namespace traj
}  // namespace finalicp