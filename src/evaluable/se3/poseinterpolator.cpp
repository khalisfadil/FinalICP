#include <evaluable/se3/poseinterpolator.hpp>

#include <evaluable/se3/evaluables.hpp>
#include <evaluable/vspace/evaluables.hpp>

namespace finalicp {
    namespace se3 {
        PoseInterpolator::Ptr PoseInterpolator::MakeShared(const Time& time, const Evaluable<InType>::ConstPtr& transform1,
                                                            const Time& time1, const Evaluable<InType>::ConstPtr& transform2,
                                                            const Time& time2) {
            return std::make_shared<PoseInterpolator>(time, transform1, time1, transform2,time2);
        }

        PoseInterpolator::PoseInterpolator(const Time& time, const Evaluable<InType>::ConstPtr& transform1,
                                            const Time& time1, const Evaluable<InType>::ConstPtr& transform2,
                                            const Time& time2)
            : transform1_(transform1), transform2_(transform2) {
            alpha_ = (time - time1).seconds() / (time2 - time1).seconds();
            // Calculate Faulhaber coefficients
            faulhaber_coeffs_.push_back(alpha_);
            faulhaber_coeffs_.push_back(alpha_ * (alpha_ - 1) / 2);
            faulhaber_coeffs_.push_back(alpha_ * (alpha_ - 1) * (2 * alpha_ - 1) / 12);
            faulhaber_coeffs_.push_back(alpha_ * alpha_ * (alpha_ - 1) * (alpha_ - 1) /
                                        24);
        }

        bool PoseInterpolator::active() const {
            return transform1_->active() || transform2_->active();
        }

        void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
            transform1_->getRelatedVarKeys(keys);
            transform2_->getRelatedVarKeys(keys);
        }

        auto PoseInterpolator::value() const -> OutType {
            const auto T1 = transform1_->value();
            const auto T2 = transform2_->value();
            const Eigen::Matrix<double, 6, 1> xi_i1 = alpha_ * (T2 / T1).vec();
            return math::se3::Transformation(xi_i1) * T1;
        }

        auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
            const auto T1 = transform1_->forward();
            const auto T2 = transform2_->forward();
            const Eigen::Matrix<double, 6, 1> xi_i1 =
                alpha_ * (T2->value() / T1->value()).vec();
            // Calculate interpolated relative transformation matrix
            const math::se3::Transformation T_i1(xi_i1);
            OutType T_i0 = T_i1 * T1->value();
            const auto node = Node<OutType>::MakeShared(T_i0);
            node->addChild(T1);
            node->addChild(T2);
            return node;
        }

        void PoseInterpolator::backward(const Eigen::MatrixXd& lhs,
                                        const Node<OutType>::Ptr& node,
                                        Jacobians& jacs) const {
            if (!active()) return;
            const auto T1 = transform1_->value();
            const auto T2 = transform2_->value();
            // Get se3 algebra of relative matrix
            const Eigen::Matrix<double, 6, 6> xi_21_curlyhat =
                math::se3::curlyhat((T2 / T1).vec());

            Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 6> xictmp = Eigen::Matrix<double, 6, 6>::Identity();
            for (size_t i = 0; i < faulhaber_coeffs_.size(); i++) {
                if (i > 0) xictmp = xi_21_curlyhat * xictmp;
                A += faulhaber_coeffs_[i] * xictmp;
            }

            if (transform1_->active()) {
                const auto T1_ = std::static_pointer_cast<Node<InType>>(node->at(0));
                const Eigen::Matrix<double, 6, 6> jac =
                    (Eigen::Matrix<double, 6, 6>::Identity() - A);
                transform1_->backward(lhs * jac, T1_, jacs);
            }
            if (transform2_->active()) {
                const auto T2_ = std::static_pointer_cast<Node<InType>>(node->at(1));
                transform2_->backward(lhs * A, T2_, jacs);
            }
        }
    }  // namespace se3
}  // namespace finalicp