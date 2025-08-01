#include <trajectory/constacc/poseinterpolator.hpp>

#include <evaluable/se3/evaluables.hpp>
#include <evaluable/vspace/evaluables.hpp>
#include <trajectory/constacc/evaluable/composecurlyhatevaluator.hpp>
#include <trajectory/constacc/helper.hpp>
#include <trajectory/constvel/evaluable/jinvvelocityevaluator.hpp>

#include<iostream>

namespace finalicp {
    namespace traj {
        namespace const_acc {

            // ###########################################################
            // MakeShared
            // ###########################################################

            PoseInterpolator::Ptr PoseInterpolator::MakeShared(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
                return std::make_shared<PoseInterpolator>(time, knot1, knot2);
            }
            
            // ###########################################################
            // PoseInterpolator
            // ###########################################################

            PoseInterpolator::PoseInterpolator(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2)
                    : knot1_(knot1), knot2_(knot2) {
                // Calculate time constants
                const double T = (knot2->time() - knot1->time()).seconds();
                const double tau = (time - knot1->time()).seconds();
#ifdef DEBUG
                // std::cout << " interpolating pose with const_acc model. Interval T: " << T << "s, at tau: " << tau << "s." << std::endl;
                if (T <= 0) {
                    std::cerr << "[CONSTACC PoseInterpolator DEBUG] CRITICAL: Total time interval T is zero or negative!" << std::endl;
                }
#endif
                const double kappa = (knot2->time() - time).seconds();
                // Q and Transition matrix
                const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
                const auto Q_tau = getQ(tau, ones);
                const auto Qinv_T = getQinv(T, ones);
                const auto Tran_kappa = getTran(kappa);
                const auto Tran_tau = getTran(tau);
                const auto Tran_T = getTran(T);

                // Calculate interpolation values
                omega_ = (Q_tau * Tran_kappa.transpose() * Qinv_T);
                lambda_ = (Tran_tau - omega_ * Tran_T);
#ifdef DEBUG
                if (!omega_.allFinite() || !lambda_.allFinite()) {
                    std::cerr << "[CONSTACC PoseInterpolator DEBUG] CRITICAL: Final interpolation matrices (omega/lambda) contain non-finite values!" << std::endl;
                }
#endif

            }

            // ###########################################################
            // active
            // ###########################################################

            bool PoseInterpolator::active() const {
                return knot1_->pose()->active() || knot1_->velocity()->active() ||knot1_->acceleration()->active() || knot2_->pose()->active() ||knot2_->velocity()->active() || knot2_->acceleration()->active();
            }

            // ###########################################################
            // getRelatedVarKeys
            // ###########################################################

            void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
                knot1_->pose()->getRelatedVarKeys(keys);
                knot1_->velocity()->getRelatedVarKeys(keys);
                knot1_->acceleration()->getRelatedVarKeys(keys);
                knot2_->pose()->getRelatedVarKeys(keys);
                knot2_->velocity()->getRelatedVarKeys(keys);
                knot2_->acceleration()->getRelatedVarKeys(keys);
            }

            // ###########################################################
            // value
            // ###########################################################

            auto PoseInterpolator::value() const -> OutType {
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();
#ifdef DEBUG
                if (!w1.allFinite() || !dw1.allFinite() || !w2.allFinite() || !dw2.allFinite()) {
                    std::cerr << "[CONSTACC PoseInterpolator DEBUG |  value] CRITICAL: Input knot states are non-finite!" << std::endl;
                }
#endif
                // Get se3 algebra of relative matrix
                const auto xi_21 = (T2 / T1).vec();
                // Calculate the 6x6 associated Jacobian
                const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
                // Calculate interpolated relative se3 algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda_.block<6, 6>(0, 6) * w1 + lambda_.block<6, 6>(0, 12) * dw1 + omega_.block<6, 6>(0, 0) * xi_21 + omega_.block<6, 6>(0, 6) * J_21_inv * w2 + omega_.block<6, 6>(0, 12) * (-0.5 * math::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);
#ifdef DEBUG
                if (!xi_i1.allFinite()) {
                    std::cerr << "[CONSTACC PoseInterpolator DEBUG | value] CRITICAL: Intermediate xi_i1 vector is non-finite!" << std::endl;
                }
#endif
                // Calculate interpolated relative transformation matrix
                const math::se3::Transformation T_i1(xi_i1);
                OutType T_i0 = T_i1 * T1;
#ifdef DEBUG
                if (!T_i0.matrix().allFinite()) {
                    std::cerr << "[CONSTACC PoseInterpolator DEBUG | value] CRITICAL: Final interpolated pose is non-finite!" << std::endl;
                }
#endif
                return T_i0;
            }

            // ###########################################################
            // forward
            // ###########################################################

            auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
                const auto T1 = knot1_->pose()->forward();
                const auto w1 = knot1_->velocity()->forward();
                const auto dw1 = knot1_->acceleration()->forward();
                const auto T2 = knot2_->pose()->forward();
                const auto w2 = knot2_->velocity()->forward();
                const auto dw2 = knot2_->acceleration()->forward();
                // Get se3 algebra of relative matrix
                const auto xi_21 = (T2->value() / T1->value()).vec();
                // Calculate the 6x6 associated Jacobian
                const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
                // Calculate interpolated relative se3 algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda_.block<6, 6>(0, 6) * w1->value() + lambda_.block<6, 6>(0, 12) * dw1->value() + omega_.block<6, 6>(0, 0) * xi_21 + omega_.block<6, 6>(0, 6) * J_21_inv * w2->value() + omega_.block<6, 6>(0, 12) *(-0.5 * math::se3::curlyhat(J_21_inv * w2->value()) * w2->value() + J_21_inv * dw2->value());
                // Calculate interpolated relative transformation matrix
                const math::se3::Transformation T_i1(xi_i1);
                OutType T_i0 = T_i1 * T1->value();
                const auto node = Node<OutType>::MakeShared(T_i0);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(dw1);
                node->addChild(T2);
                node->addChild(w2);
                node->addChild(dw2);
                return node;
            }

            // ###########################################################
            // backward
            // ###########################################################

            void PoseInterpolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                if (!active()) return;
#ifdef DEBUG
                if (!lhs.allFinite()) {
                    std::cerr << "[CONSTACC PoseInterpolator DEBUG | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                }
#endif
                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();
                // Get se3 algebra of relative matrix
                const auto xi_21 = (T2 / T1).vec();
                // Calculate the 6x6 associated Jacobian
                const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
                // Calculate interpolated relative se3 algebra
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda_.block<6, 6>(0, 6) * w1 + lambda_.block<6, 6>(0, 12) * dw1 + omega_.block<6, 6>(0, 0) * xi_21 + omega_.block<6, 6>(0, 6) * J_21_inv * w2 + omega_.block<6, 6>(0, 12) * (-0.5 * math::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);
                // Calculate interpolated relative transformation matrix
                const math::se3::Transformation T_21(xi_21);
                const math::se3::Transformation T_i1(xi_i1);
                // Calculate the 6x6 Jacobian associated with the interpolated relative
                // transformation matrix
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);

                if (knot1_->pose()->active() || knot2_->pose()->active()) {
                    // Precompute part of jacobian matrices
                    const Eigen::Matrix<double, 6, 6> w = J_i1 * (omega_.block<6, 6>(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega_.block<6, 6>(0, 6) * 0.5 * math::se3::curlyhat(w2) + omega_.block<6, 6>(0, 12) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega_.block<6, 6>(0, 12) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv;
                    if (knot1_->pose()->active()) {
                        const auto T1_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
                        Eigen::MatrixXd new_lhs = lhs * (-w * T_21.adjoint() + T_i1.adjoint());
#ifdef DEBUG
                        if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseInterpolator DEBUG | backward] CRITICAL: Derivative to T1 is non-finite!" << std::endl;
#endif
                        knot1_->pose()->backward(new_lhs, T1_, jacs);
                    }
                    if (knot2_->pose()->active()) {
                        const auto T2_ = std::static_pointer_cast<Node<InPoseType>>(node->at(3));
                        Eigen::MatrixXd new_lhs = lhs * w;
#ifdef DEBUG
                        if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseInterpolator DEBUG | backward] CRITICAL: Derivative to T2 is non-finite!" << std::endl;
#endif
                        knot2_->pose()->backward(new_lhs, T2_, jacs);
                    }
                }
                if (knot1_->velocity()->active()) {
                    const auto w1_ = std::static_pointer_cast<Node<InVelType>>(node->at(1));
                    Eigen::MatrixXd new_lhs = lhs * lambda_.block<6, 6>(0, 6) * J_i1;
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseInterpolator DEBUG | backward] CRITICAL: Derivative to w1 is non-finite!" << std::endl;
#endif
                    knot1_->velocity()->backward(new_lhs, w1_, jacs);
                }
                if (knot2_->velocity()->active()) {
                    const auto w2_ = std::static_pointer_cast<Node<InVelType>>(node->at(4));
                    Eigen::MatrixXd new_lhs = lhs * (omega_.block<6, 6>(0, 6) * J_i1 * J_21_inv + omega_.block<6, 6>(0, 12) * -0.5 * J_i1 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv));
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseInterpolator DEBUG | backward] CRITICAL: Derivative to w2 is non-finite!" << std::endl;
#endif
                    knot2_->velocity()->backward(new_lhs, w2_, jacs);
                }
                if (knot1_->acceleration()->active()) {
                    const auto dw1_ = std::static_pointer_cast<Node<InAccType>>(node->at(2));
                    Eigen::MatrixXd new_lhs = lhs * lambda_.block<6, 6>(0, 12) * J_i1;
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseInterpolator DEBUG | backward] CRITICAL: Derivative to dw1 is non-finite!" << std::endl;
#endif
                    knot1_->acceleration()->backward(new_lhs, dw1_, jacs);
                }
                if (knot2_->acceleration()->active()) {
                    const auto dw2_ = std::static_pointer_cast<Node<InAccType>>(node->at(5));
                    Eigen::MatrixXd new_lhs = lhs * omega_.block<6, 6>(0, 12) * J_i1 * J_21_inv;
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PoseInterpolator DEBUG | backward] CRITICAL: Derivative to dw2 is non-finite!" << std::endl;
#endif
                    knot2_->acceleration()->backward(new_lhs, dw2_, jacs);
                }
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace finalicp