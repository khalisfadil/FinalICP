#include <trajectory/constvel/velocityinterpolator.hpp>
#include <iostream>

#include <evaluable/se3/evaluables.hpp>
#include <evaluable/vspace/evaluables.hpp>
#include <trajectory/constvel/evaluable/jvelocityevaluator.hpp>
#include <trajectory/constvel/evaluable/jinvvelocityevaluator.hpp>
#include <trajectory/constvel/helper.hpp>

namespace finalicp {
    namespace traj {
        namespace const_vel {

            VelocityInterpolator::Ptr VelocityInterpolator::MakeShared(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
                return std::make_shared<VelocityInterpolator>(time, knot1, knot2);
            }

            VelocityInterpolator::VelocityInterpolator(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2)
                    : knot1_(knot1), knot2_(knot2) {
                // Calculate time constants
                const double T = (knot2->time() - knot1->time()).seconds();
                const double tau = (time - knot1->time()).seconds();
                const double kappa = (knot2->time() - time).seconds();
                const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
                const auto Q_tau = getQ(tau, ones);
                const auto Qinv_T = getQinv(T, ones);
                const auto Tran_kappa = getTran(kappa);
                const auto Tran_tau = getTran(tau);
                const auto Tran_T = getTran(T);
                // Calculate interpolation values
                const auto psi = (Q_tau * Tran_kappa.transpose() * Qinv_T);
                const auto lambda = (Tran_tau - psi * Tran_T);

                // const double ratio = tau / T;
                // const double ratio2 = ratio * ratio;
                // const double ratio3 = ratio2 * ratio;
                // // Calculate 'psi' interpolation values
                psi11_ = psi(0, 0);
                psi12_ = psi(0, 6);
                psi21_ = psi(6, 0);
                psi22_ = psi(6, 6);
                lambda11_ = lambda(0, 0);
                lambda12_ = lambda(0, 6);
                lambda21_ = lambda(6, 0);
                lambda22_ = lambda(6, 6);
                // psi11_ = 3.0 * ratio2 - 2.0 * ratio3;
                // psi12_ = tau * (ratio2 - ratio);
                // psi21_ = 6.0 * (ratio - ratio2) / T;
                // psi22_ = 3.0 * ratio2 - 2.0 * ratio;
                // // Calculate 'lambda' interpolation values
                // lambda11_ = 1.0 - psi11_;
                // lambda12_ = tau - T * psi11_ - psi12_;
                // lambda21_ = -psi21_;
                // lambda22_ = 1.0 - T * psi21_ - psi22_;
            }

            bool VelocityInterpolator::active() const {
                return knot1_->pose()->active() || knot1_->velocity()->active() || knot2_->pose()->active() || knot2_->velocity()->active();
            }

            void VelocityInterpolator::getRelatedVarKeys(KeySet& keys) const {
                knot1_->pose()->getRelatedVarKeys(keys);
                knot1_->velocity()->getRelatedVarKeys(keys);
                knot2_->pose()->getRelatedVarKeys(keys);
                knot2_->velocity()->getRelatedVarKeys(keys);
            }

            auto VelocityInterpolator::value() const -> OutType {
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
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 1> xi_j1 = lambda22_ * w1 + psi21_ * xi_21 + psi22_ * J_21_inv * w2;
                // Calculate interpolated body-centric velocity
                OutType xi_it = J_i1 * xi_j1;
                return xi_it;
            }

            auto VelocityInterpolator::forward() const -> Node<OutType>::Ptr {
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
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 1> xi_j1 = lambda22_ * w1->value() + psi21_ * xi_21 + psi22_ * J_21_inv * w2->value();
                // Calculate interpolated body-centric velocity
                OutType xi_it = J_i1 * xi_j1;
                const auto node = Node<OutType>::MakeShared(xi_it);
                node->addChild(T1);
                node->addChild(w1);
                node->addChild(T2);
                node->addChild(w2);
                return node;
            }

            void VelocityInterpolator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                if (!active()) return;
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
                const Eigen::Matrix<double, 6, 1> xi_j1 = lambda22_ * w1 + psi21_ * xi_21 + psi22_ * J_21_inv * w2;
                // Calculate the 6x6 Jacobian associated with the interpolated relative
                // transformation matrix
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                // temp value for speed
                // const auto xi_j1_c = lgmath::se3::curlyhat(xi_j1);
                // const auto xi_i1_c = lgmath::se3::curlyhat(xi_i1);
                const Eigen::Matrix<double, 6, 6> xi_j1_ch = -0.5 * math::se3::curlyhat(xi_j1);
                // (1 / 6) * lgmath::se3::curlyhat(xi_i1_c * xi_j1) -
                // (1 / 6) * xi_i1_c * xi_j1_c -
                // (1 / 24) * lgmath::se3::curlyhat(xi_i1_c * xi_i1_c * xi_j1) -
                // (1 / 24) * xi_i1_c * lgmath::se3::curlyhat(xi_i1_c * xi_j1) -
                // (1 / 24) * xi_i1_c * xi_i1_c * xi_j1_c;

                // Calculate relative transformation matrix
                // const lgmath::se3::Transformation T_21(xi_21);
                const math::se3::Transformation T_21 = T2 * T1.inverse();
                if (knot1_->pose()->active() || knot2_->pose()->active()) {
                    // Precompute part of jacobian matrices
                    const Eigen::Matrix<double, 6, 6> wtmp = (0.5 * math::se3::curlyhat(w2) ) * J_21_inv;
                    const Eigen::Matrix<double, 6, 6> w = J_i1 * (psi21_ * J_21_inv + psi22_ * wtmp) + xi_j1_ch * (psi11_ * J_21_inv + psi12_ * wtmp);
                    if (knot1_->pose()->active()) {
                        const auto T1_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
                        Eigen::MatrixXd new_lhs = lhs * (-w * T_21.adjoint());
                        knot1_->pose()->backward(new_lhs, T1_, jacs);
                    }
                    if (knot2_->pose()->active()) {
                        const auto T2_ = std::static_pointer_cast<Node<InPoseType>>(node->at(2));
                        Eigen::MatrixXd new_lhs = lhs * w;
                        knot2_->pose()->backward(new_lhs, T2_, jacs);
                    }
                }
                if (knot1_->velocity()->active()) {
                    const auto w1_ = std::static_pointer_cast<Node<InVelType>>(node->at(1));
                    Eigen::MatrixXd new_lhs = lhs * (J_i1 * lambda22_ + xi_j1_ch * lambda12_);
                    knot1_->velocity()->backward(new_lhs, w1_, jacs);
                }
                if (knot2_->velocity()->active()) {
                    const auto w2_ = std::static_pointer_cast<Node<InVelType>>(node->at(3));
                    Eigen::MatrixXd new_lhs = lhs * (J_i1 * (psi22_ * J_21_inv) + xi_j1_ch * (psi12_ * J_21_inv));
                    knot2_->velocity()->backward(new_lhs, w2_, jacs);
                }
            }
        }  // namespace const_vels
    }  // namespace traj
}  // namespace finalicp