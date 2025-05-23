#include <evaluable/imu/gyroerrorevaluator.hpp>

#include <iostream>

#include <so3/operations.hpp>


namespace finalicp {
    namespace imu {

        auto GyroErrorEvaluator::MakeShared(const Evaluable<VelInType>::ConstPtr &velocity, const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
                -> Ptr {
            return std::make_shared<GyroErrorEvaluator>(velocity, bias, gyro_meas);
        }

        GyroErrorEvaluator::GyroErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity, const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
                : velocity_(velocity), bias_(bias), gyro_meas_(gyro_meas) {
            jac_vel_.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
            jac_bias_.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity() * -1;
        }

        bool GyroErrorEvaluator::active() const {
            return velocity_->active() || bias_->active();
        }

        void GyroErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
            velocity_->getRelatedVarKeys(keys);
            bias_->getRelatedVarKeys(keys);
        }

        auto GyroErrorEvaluator::value() const -> OutType {
            OutType error = gyro_meas_ + velocity_->value().block<3, 1>(3, 0) - bias_->value().block<3, 1>(3, 0);
            return error;
        }

        auto GyroErrorEvaluator::forward() const -> Node<OutType>::Ptr {
            const auto child1 = velocity_->forward();
            const auto child2 = bias_->forward();
            const auto w_mv_in_v = child1->value();
            const auto b = child2->value();

            // clang-format off
            OutType error = gyro_meas_ + w_mv_in_v.block<3, 1>(3, 0) - b.block<3, 1>(3, 0);
            // clang-format on

            const auto node = Node<OutType>::MakeShared(error);
            node->addChild(child1);
            node->addChild(child2);

            return node;
        }

        void GyroErrorEvaluator::backward(const Eigen::MatrixXd &lhs,const Node<OutType>::Ptr &node, Jacobians &jacs) const {
            const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->at(0));
            const auto child2 = std::static_pointer_cast<Node<BiasInType>>(node->at(1));

            if (velocity_->active()) {
                velocity_->backward(lhs * jac_vel_, child1, jacs);
            }
            if (bias_->active()) {
                bias_->backward(lhs * jac_bias_, child2, jacs);
            }
        }

        void GyroErrorEvaluator::getMeasJacobians(JacType &jac_vel, JacType &jac_bias) const {
            jac_vel = jac_vel_;
            jac_bias = jac_bias_;
        }

        GyroErrorEvaluator::Ptr GyroError(const Evaluable<GyroErrorEvaluator::VelInType>::ConstPtr &velocity,
                                            const Evaluable<GyroErrorEvaluator::BiasInType>::ConstPtr &bias,
                                            const GyroErrorEvaluator::ImuInType &gyro_meas) {
            return GyroErrorEvaluator::MakeShared(velocity, bias, gyro_meas);
        }
    }  // namespace imu
}  // namespace finalicp

