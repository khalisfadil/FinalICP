#include <evaluable/imu/accerrorevaluator.hpp>

#include <iostream>

#include <so3/operations.hpp>


namespace finalicp {
    namespace imu {

        auto AccelerationErrorEvaluator::MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                                                        const Evaluable<AccInType>::ConstPtr &acceleration,
                                                        const Evaluable<BiasInType>::ConstPtr &bias,
                                                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                                        const ImuInType &acc_meas) -> Ptr {
                return std::make_shared<AccelerationErrorEvaluator>(transform, acceleration, bias, transform_i_to_m, acc_meas);
        }

        AccelerationErrorEvaluator::AccelerationErrorEvaluator(const Evaluable<PoseInType>::ConstPtr &transform,
                                                                const Evaluable<AccInType>::ConstPtr &acceleration,
                                                                const Evaluable<BiasInType>::ConstPtr &bias,
                                                                const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                                                                const ImuInType &acc_meas)
                : transform_(transform),
                acceleration_(acceleration),
                bias_(bias),
                transform_i_to_m_(transform_i_to_m),
                acc_meas_(acc_meas) {
            gravity_(2, 0) = -9.8042;
            jac_accel_.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();
            jac_bias_.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * -1;
        }

        bool AccelerationErrorEvaluator::active() const {
            return transform_->active() || acceleration_->active() || bias_->active() || transform_i_to_m_->active();
        }

        void AccelerationErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
            transform_->getRelatedVarKeys(keys);
            acceleration_->getRelatedVarKeys(keys);
            bias_->getRelatedVarKeys(keys);
            transform_i_to_m_->getRelatedVarKeys(keys);
        }

        auto AccelerationErrorEvaluator::value() const -> OutType {

            const Eigen::Matrix3d C_vm = transform_->value().C_ba();
            const Eigen::Matrix3d C_mi = transform_i_to_m_->value().C_ba();
            OutType error = acc_meas_ + acceleration_->value().block<3, 1>(0, 0) + C_vm * C_mi * gravity_ - bias_->value().block<3, 1>(0, 0);
            return error;

        }

        auto AccelerationErrorEvaluator::forward() const -> Node<OutType>::Ptr {
            const auto child1 = transform_->forward();
            const auto child2 = acceleration_->forward();
            const auto child3 = bias_->forward();
            const auto child4 = transform_i_to_m_->forward();

            const auto C_vm = child1->value().C_ba();
            const auto dw_mv_in_v = child2->value();
            const auto b = child3->value();
            const auto C_mi = child4->value().C_ba();

            // clang-format off
            OutType error = acc_meas_.block<3, 1>(0, 0) + dw_mv_in_v.block<3, 1>(0, 0) + C_vm * C_mi * gravity_ - b.block<3, 1>(0, 0);
            // clang-format on

            const auto node = Node<OutType>::MakeShared(error);
            node->addChild(child1);
            node->addChild(child2);
            node->addChild(child3);
            node->addChild(child4);

            return node;
        }

        void AccelerationErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                          const Node<OutType>::Ptr &node,
                                          Jacobians &jacs) const {
            const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
            const auto child2 = std::static_pointer_cast<Node<AccInType>>(node->at(1));
            const auto child3 = std::static_pointer_cast<Node<BiasInType>>(node->at(2));
            const auto child4 = std::static_pointer_cast<Node<PoseInType>>(node->at(3));

            if (transform_->active()) {
                JacType jac = JacType::Zero();
                jac.block<3, 3>(0, 3) = -1 * math::so3::hat(child1->value().C_ba() * child4->value().C_ba() * gravity_);
                transform_->backward(lhs * jac, child1, jacs);
            }

            if (acceleration_->active()) {
                acceleration_->backward(lhs * jac_accel_, child2, jacs);
            }

            if (bias_->active()) {
                bias_->backward(lhs * jac_bias_, child3, jacs);
            }

            if (transform_i_to_m_->active()) {
                JacType jac = JacType::Zero();
                jac.block<3, 3>(0, 3) = -1 * child1->value().C_ba() * math::so3::hat(child4->value().C_ba() * gravity_);
                transform_i_to_m_->backward(lhs * jac, child4, jacs);
            }
            
        }

        void AccelerationErrorEvaluator::getMeasJacobians(JacType &jac_pose,
                                                  JacType &jac_accel,
                                                  JacType &jac_bias,
                                                  JacType &jac_T_mi) const {
            const Eigen::Matrix3d C_vm = transform_->value().C_ba();
            const Eigen::Matrix3d C_mi = transform_i_to_m_->value().C_ba();
            jac_pose.setZero();
            jac_pose.block<3, 3>(0, 3) = -1 * math::so3::hat(C_vm * C_mi * gravity_);
            jac_accel.setZero();
            jac_accel.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();
            jac_bias.setZero();
            jac_bias.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * -1;
            jac_T_mi.setZero();
            jac_T_mi.block<3, 3>(0, 3) = -1 * C_vm * math::so3::hat(C_mi * gravity_);
        }
        
        AccelerationErrorEvaluator::Ptr AccelerationError(
            const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform,
            const Evaluable<AccelerationErrorEvaluator::AccInType>::ConstPtr &acceleration,
            const Evaluable<AccelerationErrorEvaluator::BiasInType>::ConstPtr &bias,
            const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
            const AccelerationErrorEvaluator::ImuInType &acc_meas) {
            return AccelerationErrorEvaluator::MakeShared(transform, acceleration, bias, transform_i_to_m, acc_meas);
        }
    }  // namespace imu
}  // namespace finalicp

