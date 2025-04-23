#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace imu {
        class AccelerationErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
            public:
                using Ptr = std::shared_ptr<AccelerationErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const AccelerationErrorEvaluator>;

                using PoseInType = math::se3::Transformation;
                using AccInType = Eigen::Matrix<double, 6, 1>;
                using BiasInType = Eigen::Matrix<double, 6, 1>;
                using ImuInType = Eigen::Matrix<double, 3, 1>;
                using OutType = Eigen::Matrix<double, 3, 1>;
                using Time = traj::Time;
                using JacType = Eigen::Matrix<double, 3, 6>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                        const Evaluable<AccInType>::ConstPtr &acceleration,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                        const ImuInType &acc_meas);

                //Constructor.
                AccelerationErrorEvaluator(
                    const Evaluable<PoseInType>::ConstPtr &transform,
                    const Evaluable<AccInType>::ConstPtr &acceleration,
                    const Evaluable<BiasInType>::ConstPtr &bias,
                    const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                    const ImuInType &acc_meas);

                //Checks if the acceleration error is influenced by active state variables.
                bool active() const override;

                //Collects state variable keys that influence this evaluator.
                void getRelatedVarKeys(KeySet &keys) const override;

                //Computes the acceleration error.
                OutType value() const override;

                //Forward evaluation of acceleration error.
                Node<OutType>::Ptr forward() const override;

                //Computes Jacobians for the acceleration error.
                void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,Jacobians &jacs) const override;

                //Sets the gravity vector.
                void setGravity(double gravity) { gravity_(2, 0) = gravity; }

                //Sets the timestamp for the acceleration measurement.
                void setTime(Time time) {
                    time_ = time;
                    time_init_ = true;
                };

                //Retrieves the timestamp for the acceleration measurement.
                Time getTime() const {
                    if (time_init_)
                    return time_;
                    else
                    throw std::runtime_error("Accel measurement time was not initialized");
                }

                //Computes measurement Jacobians.
                void getMeasJacobians(JacType &jac_pose, JacType &jac_accel,
                        JacType &jac_bias, JacType &jac_T_mi) const;

            private:
                const Evaluable<PoseInType>::ConstPtr transform_;          //Transformation state.
                const Evaluable<AccInType>::ConstPtr acceleration_;        //Acceleration state.
                const Evaluable<BiasInType>::ConstPtr bias_;              //IMU bias.
                const Evaluable<PoseInType>::ConstPtr transform_i_to_m_;  //IMU-to-measurement transformation.
                const ImuInType acc_meas_;                                //Measured acceleration.

                JacType jac_accel_ = JacType::Zero();                     //Acceleration Jacobian.
                JacType jac_bias_ = JacType::Zero();                      //Bias Jacobian.
                Eigen::Matrix<double, 3, 1> gravity_ = Eigen::Matrix<double, 3, 1>::Zero(); //Gravity vector.
                Time time_;                                               //Timestamp of measurement.
                bool time_init_ = false;                                  //Flag indicating if time was set.

        };

        AccelerationErrorEvaluator::Ptr AccelerationError(const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform,
                    const Evaluable<AccelerationErrorEvaluator::AccInType>::ConstPtr &acceleration,
                    const Evaluable<AccelerationErrorEvaluator::BiasInType>::ConstPtr &bias,
                    const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
                    const AccelerationErrorEvaluator::ImuInType &acc_meas);


    }  // namespace imu
}  // namespace finalicp