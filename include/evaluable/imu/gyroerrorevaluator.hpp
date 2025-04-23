#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace imu {
        class GyroErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
            public:
                using Ptr = std::shared_ptr<GyroErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const GyroErrorEvaluator>;

                using VelInType = Eigen::Matrix<double, 6, 1>;
                using BiasInType = Eigen::Matrix<double, 6, 1>;
                using ImuInType = Eigen::Matrix<double, 3, 1>;
                using OutType = Eigen::Matrix<double, 3, 1>;
                using Time = traj::Time;
                using JacType = Eigen::Matrix<double, 3, 6>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const ImuInType &gyro_meas);

                //Constructor.
                GyroErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
                                    const Evaluable<BiasInType>::ConstPtr &bias,
                                    const ImuInType &gyro_meas);

                //Checks if the DMI error is influenced by active state variables.
                bool active() const override;

                //Collects state variable keys that influence this evaluator.
                void getRelatedVarKeys(KeySet &keys) const override;

                //Computes the DMI error.
                OutType value() const override;

                //Forward evaluation of DMI error.
                Node<OutType>::Ptr forward() const override;

                //Computes Jacobians for the DMI error.
                void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node, Jacobians &jacs) const override;

                //Sets the timestamp for the DMI measurement.
                void setTime(Time time) {
                    time_ = time;
                    time_init_ = true;
                };

                //Retrieves the timestamp for the DMI measurement.
                Time getTime() const {
                    if (time_init_)
                    return time_;
                    else
                    throw std::runtime_error("Gyro measurement time was not initialized");
                }

                //Computes measurement Jacobians.
                void getMeasJacobians(JacType &jac_vel, JacType &jac_bias) const;
                
            private:
                const Evaluable<VelInType>::ConstPtr velocity_;  ///< Estimated velocity state.
                const Evaluable<BiasInType>::ConstPtr bias_;    ///< Estimated IMU bias.
                const ImuInType gyro_meas_;                     ///< Measured gyroscope data.

                JacType jac_vel_ = JacType::Zero();             ///< Velocity Jacobian.
                JacType jac_bias_ = JacType::Zero();            ///< Bias Jacobian.

                Time time_;                                     ///< Timestamp of measurement.
                bool time_init_ = false;                        ///< Flag indicating if time was set.
        };

        //Factory function for creating a GyroErrorEvaluator.
        GyroErrorEvaluator::Ptr GyroError(const Evaluable<GyroErrorEvaluator::VelInType>::ConstPtr &velocity,
                                            const Evaluable<GyroErrorEvaluator::BiasInType>::ConstPtr &bias,
                                            const GyroErrorEvaluator::ImuInType &gyro_meas);
    }  // namespace imu
}  // namespace finalicp
