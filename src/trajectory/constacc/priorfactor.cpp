#include <trajectory/constacc/priorfactor.hpp>
#include <trajectory/constacc/helper.hpp>

#include <evaluable/se3/evaluables.hpp>
#include <evaluable/vspace/evaluables.hpp>
#include <trajectory/constacc/evaluable/composecurlyhatevaluator.hpp>
#include <trajectory/constvel/evaluable/jinvvelocityevaluator.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto PriorFactor::MakeShared(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) -> Ptr {
                return std::make_shared<PriorFactor>(knot1, knot2);
            }

            // ###########################################################
            // PriorFactor
            // ###########################################################

            PriorFactor::PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2)
                : knot1_(knot1), knot2_(knot2) {
                const double dt = (knot2_->time() - knot1_->time()).seconds();
                Phi_ = getTran(dt);
#ifdef DEBUG
                // --- [IMPROVEMENT] Log factor creation and sanity check the transition matrix ---
                std::cout << "[CONSTACC PriorFactor DEBUG] Creating motion factor between knots at t="
                        << std::fixed << knot1_->time().seconds() << " and t=" << knot2_->time().seconds()
                        << " (dt=" << dt << "s)." << std::endl;
                if (dt <= 0) {
                    std::cerr << "[CONSTACC PriorFactor DEBUG] CRITICAL: Time delta (dt) is zero or negative!" << std::endl;
                }
                if (!Phi_.allFinite()) {
                    std::cerr << "[CONSTACC PriorFactor DEBUG] CRITICAL: Computed transition matrix Phi_ contains non-finite values!" << std::endl;
                }
#endif
            }

            // ###########################################################
            // active
            // ###########################################################

            bool PriorFactor::active() const {
                return knot1_->pose()->active() || knot1_->velocity()->active() || knot1_->acceleration()->active() || knot2_->pose()->active() || knot2_->velocity()->active() || knot2_->acceleration()->active();
            }

            // ###########################################################
            // getRelatedVarKeys
            // ###########################################################

            void PriorFactor::getRelatedVarKeys(KeySet& keys) const {
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

            auto PriorFactor::value() const -> OutType {
                OutType error = OutType::Zero();

                const auto T1 = knot1_->pose()->value();
                const auto w1 = knot1_->velocity()->value();
                const auto dw1 = knot1_->acceleration()->value();
                const auto T2 = knot2_->pose()->value();
                const auto w2 = knot2_->velocity()->value();
                const auto dw2 = knot2_->acceleration()->value();

#ifdef DEBUG
                if (!w1.allFinite() || !dw1.allFinite() || !w2.allFinite() || !dw2.allFinite()) {
                    std::cerr << "[CONSTACC PriorFactor DEBUG | value] CRITICAL: Input knot states are non-finite!" << std::endl;
                }
#endif

                const auto xi_21 = (T2 / T1).vec();
                const auto J_21_inv = math::se3::vec2jacinv(xi_21);

                Eigen::Matrix<double, 18, 1> gamma1 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma1.block<6, 1>(6, 0) = w1;
                gamma1.block<6, 1>(12, 0) = dw1;
                Eigen::Matrix<double, 18, 1> gamma2 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma2.block<6, 1>(0, 0) = xi_21;
                gamma2.block<6, 1>(6, 0) = J_21_inv * w2;
                gamma2.block<6, 1>(12, 0) = -0.5 * math::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2;
                error = gamma2 - Phi_ * gamma1;
#ifdef DEBUG
                if (!gamma1.allFinite() || !gamma2.allFinite()) {
                    std::cerr << "[CONSTACC PriorFactor DEBUG | value] CRITICAL: Intermediate gamma vectors are non-finite!" << std::endl;
                }
                if (!error.allFinite()) {
                    std::cerr << "[CONSTACC PriorFactor DEBUG | value] CRITICAL: Final error vector is non-finite!" << std::endl;
                }
#endif
                return error;
            }

            // ###########################################################
            // forward
            // ###########################################################

            auto PriorFactor::forward() const -> Node<OutType>::Ptr {
                const auto T1 = knot1_->pose()->forward();
                const auto w1 = knot1_->velocity()->forward();
                const auto dw1 = knot1_->acceleration()->forward();
                const auto T2 = knot2_->pose()->forward();
                const auto w2 = knot2_->velocity()->forward();
                const auto dw2 = knot2_->acceleration()->forward();

                const auto xi_21 = (T2->value() / T1->value()).vec();
                const auto J_21_inv = math::se3::vec2jacinv(xi_21);
                OutType error = OutType::Zero();
                Eigen::Matrix<double, 18, 1> gamma1 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma1.block<6, 1>(6, 0) = w1->value();
                gamma1.block<6, 1>(12, 0) = dw1->value();
                Eigen::Matrix<double, 18, 1> gamma2 = Eigen::Matrix<double, 18, 1>::Zero();
                gamma2.block<6, 1>(0, 0) = xi_21;
                gamma2.block<6, 1>(6, 0) = J_21_inv * w2->value();
                gamma2.block<6, 1>(12, 0) = -0.5 * math::se3::curlyhat(J_21_inv * w2->value()) * w2->value() + J_21_inv * dw2->value();
                error = gamma2 - Phi_ * gamma1;
                const auto node = Node<OutType>::MakeShared(error);
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

            void PriorFactor::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
#ifdef DEBUG
                if (!lhs.allFinite()) {
                    std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                }
#endif
                if (knot1_->pose()->active() || knot1_->velocity()->active() || knot1_->acceleration()->active()) {
                    const auto Fk1 = getJacKnot1_();
#ifdef DEBUG
                    if (!Fk1.allFinite()) {
                        std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Pre-computed Jacobians Fk1 are non-finite!" << std::endl;
                    }
#endif
                    if (knot1_->pose()->active()) {
                        const auto T1 = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
                        Eigen::MatrixXd new_lhs = lhs * Fk1.block<18, 6>(0, 0);
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Derivative to T1 is non-finite!" << std::endl;
#endif
                        knot1_->pose()->backward(new_lhs, T1, jacs);
                    }
                    if (knot1_->velocity()->active()) {
                        const auto w1 = std::static_pointer_cast<Node<InVelType>>(node->at(1));
                        Eigen::MatrixXd new_lhs = lhs * Fk1.block<18, 6>(0, 6);
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Derivative to w1 is non-finite!" << std::endl;
#endif
                        knot1_->velocity()->backward(new_lhs, w1, jacs);
                    }
                    if (knot1_->acceleration()->active()) {
                        const auto dw1 = std::static_pointer_cast<Node<InAccType>>(node->at(2));
                        Eigen::MatrixXd new_lhs = lhs * Fk1.block<18, 6>(0, 12);
#ifdef DEBUG
                    if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Derivative to dw1 is non-finite!" << std::endl;
#endif
                        knot1_->acceleration()->backward(new_lhs, dw1, jacs);
                    }
                }
                if (knot2_->pose()->active() || knot2_->velocity()->active() ||
                    knot2_->acceleration()->active()) {
                    const auto Ek = getJacKnot2_();
#ifdef DEBUG
                    if (!Ek.allFinite()) {
                        std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Pre-computed Jacobians Ek are non-finite!" << std::endl;
                    }
#endif
                    if (knot2_->pose()->active()) {
                        const auto T2 = std::static_pointer_cast<Node<InPoseType>>(node->at(3));
                        Eigen::MatrixXd new_lhs = lhs * Ek.block<18, 6>(0, 0);
#ifdef DEBUG
                        if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Derivative to T2 is non-finite!" << std::endl;
#endif
                        knot2_->pose()->backward(new_lhs, T2, jacs);
                    }
                    if (knot2_->velocity()->active()) {
                        const auto w2 = std::static_pointer_cast<Node<InVelType>>(node->at(4));
                        Eigen::MatrixXd new_lhs = lhs * Ek.block<18, 6>(0, 6);
#ifdef DEBUG
                        if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Derivative to w2 is non-finite!" << std::endl;
#endif
                        knot2_->velocity()->backward(new_lhs, w2, jacs);
                    }
                    if (knot2_->acceleration()->active()) {
                        const auto dw2 = std::static_pointer_cast<Node<InAccType>>(node->at(5));
                        Eigen::MatrixXd new_lhs = lhs * Ek.block<18, 6>(0, 12);
#ifdef DEBUG
                        if (!new_lhs.allFinite()) std::cerr << "[CONSTACC PriorFactor DEBUG | backward] CRITICAL: Derivative to dw2 is non-finite!" << std::endl;
#endif
                        knot2_->acceleration()->backward(new_lhs, dw2, jacs);
                    }
                }
            }

            // ###########################################################
            // getJacKnot1_
            // ###########################################################

            Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot1_() const {
                return getJacKnot1(knot1_, knot2_);
            }

            // ###########################################################
            // getJacKnot2_
            // ###########################################################

            Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot2_() const {
                return getJacKnot2(knot1_, knot2_);
            }
        }  // namespace const_acc
    }  // namespace traj
}  // namespace finalicp