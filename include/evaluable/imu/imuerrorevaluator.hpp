#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>

namespace finalicp {
    namespace imu {
        class IMUErrorEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<IMUErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const IMUErrorEvaluator>;

                using PoseInType = math::se3::Transformation;
                using VelInType = Eigen::Matrix<double, 6, 1>;
                using AccInType = Eigen::Matrix<double, 6, 1>;
                using BiasInType = Eigen::Matrix<double, 6, 1>;
                using ImuInType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                        const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<AccInType>::ConstPtr &acceleration,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                        const ImuInType &imu_meas);

                //Constructor.
                IMUErrorEvaluator(const Evaluable<PoseInType>::ConstPtr &transform,
                        const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<AccInType>::ConstPtr &acceleration,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                        const ImuInType &imu_meas);

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

                //Sets the gravity vector.
                void setGravity(double gravity) { gravity_(2, 0) = gravity; }
                
            private:
                const Evaluable<PoseInType>::ConstPtr transform_;                       //Transformation state.
                const Evaluable<VelInType>::ConstPtr velocity_;                         //Velocity state.
                const Evaluable<AccInType>::ConstPtr acceleration_;                     //Acceleration state.
                const Evaluable<BiasInType>::ConstPtr bias_;                            //IMU bias.
                const Evaluable<PoseInType>::ConstPtr transform_i_to_m_;                //IMU-to-measurement transformation.
                const ImuInType imu_meas_;
                Eigen::Matrix<double, 3, 1> gravity_ = Eigen::Matrix<double, 3, 1>::Zero();        
        };

        //Factory function for creating a GyroErrorEvaluator.
        IMUErrorEvaluator::Ptr imuError(const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform,
                                        const Evaluable<IMUErrorEvaluator::VelInType>::ConstPtr &velocity,
                                        const Evaluable<IMUErrorEvaluator::AccInType>::ConstPtr &acceleration,
                                        const Evaluable<IMUErrorEvaluator::BiasInType>::ConstPtr &bias,
                                        const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
                                        const IMUErrorEvaluator::ImuInType &imu_meas);
    }  // namespace imu
}  // namespace finalicp
