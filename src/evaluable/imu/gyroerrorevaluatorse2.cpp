#include <evaluable/imu/gyroerrorevaluatorse2.hpp>

#include <iostream>

#include <so3/operations.hpp>


namespace finalicp {
    namespace imu {

        auto GyroErrorEvaluatorSE2::MakeShared(const Evaluable<VelInType>::ConstPtr &velocity, const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
            -> Ptr {
            return std::make_shared<GyroErrorEvaluatorSE2>(velocity, bias, gyro_meas);
        }

        GyroErrorEvaluatorSE2::GyroErrorEvaluatorSE2(const Evaluable<VelInType>::ConstPtr &velocity, const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
                : velocity_(velocity), bias_(bias), gyro_meas_(gyro_meas) {
            jac_vel_(0, 5) = 1;
            jac_bias_(0, 5) = -1;
        }

        bool GyroErrorEvaluatorSE2::active() const {
            return velocity_->active() || bias_->active();
        }

        void GyroErrorEvaluatorSE2::getRelatedVarKeys(KeySet &keys) const {
            velocity_->getRelatedVarKeys(keys);
            bias_->getRelatedVarKeys(keys);
        }

        auto GyroErrorEvaluatorSE2::value() const -> OutType {

            OutType error(gyro_meas_(2, 0) + (velocity_->value())(5, 0) - bias_->value()(5, 0));
            return error;

        }

        auto GyroErrorEvaluatorSE2::forward() const -> Node<OutType>::Ptr {
            const auto child1 = velocity_->forward();
            const auto child2 = bias_->forward();
            const auto w_mv_in_v = child1->value();
            const auto b = child2->value();

            // clang-format off
            OutType error(gyro_meas_(2, 0) + w_mv_in_v(5, 0) - b(5, 0));
            // clang-format on

            const auto node = Node<OutType>::MakeShared(error);
            node->addChild(child1);
            node->addChild(child2);

            return node;
        }

        void GyroErrorEvaluatorSE2::backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node, Jacobians &jacs) const {
            const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->at(0));
            const auto child2 = std::static_pointer_cast<Node<BiasInType>>(node->at(1));

            if (velocity_->active()) {
                velocity_->backward(lhs * jac_vel_, child1, jacs);
            }
            if (bias_->active()) {
                bias_->backward(lhs * jac_bias_, child2, jacs);
            }
            // clang-format on
        }

        GyroErrorEvaluatorSE2::Ptr GyroErrorSE2(const Evaluable<GyroErrorEvaluatorSE2::VelInType>::ConstPtr &velocity,
                                                const Evaluable<GyroErrorEvaluatorSE2::BiasInType>::ConstPtr &bias,
                                                const GyroErrorEvaluatorSE2::ImuInType &gyro_meas) {
            return GyroErrorEvaluatorSE2::MakeShared(velocity, bias, gyro_meas);
        }
    }  // namespace imu
}  // namespace finalicp

