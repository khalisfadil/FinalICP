#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace imu {
        class DMIErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
            public:
                using Ptr = std::shared_ptr<DMIErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const DMIErrorEvaluator>;

                using VelInType = Eigen::Matrix<double, 6, 1>;
                using DMIInType = double;
                using ScaleInType = Eigen::Matrix<double, 1, 1>;
                using OutType = Eigen::Matrix<double, 1, 1>;
                using Time = traj::Time;
                using JacType = Eigen::Matrix<double, 1, 6>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<ScaleInType>::ConstPtr &scale,
                        const DMIInType &dmi_meas);

                //Constructor.
                DMIErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
                    const Evaluable<ScaleInType>::ConstPtr &scale,
                    const DMIInType &dmi_meas);

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
                    throw std::runtime_error("DMI measurement time was not initialized");
                }
                
            private:
                const Evaluable<VelInType>::ConstPtr velocity_;  //Estimated velocity state.
                const Evaluable<ScaleInType>::ConstPtr scale_;  //Scale factor for DMI.
                const DMIInType dmi_meas_;                      //Measured DMI distance.

                JacType jac_vel_ = JacType::Zero();             //Velocity Jacobian.
                Eigen::Matrix<double, 1, 1> jac_scale_ = Eigen::Matrix<double, 1, 1>::Zero(); ///< Scale Jacobian.

                Time time_;                                     //Timestamp of measurement.
                bool time_init_ = false;                        //Flag indicating if time was set.
        };

        DMIErrorEvaluator::Ptr DMIError(const Evaluable<DMIErrorEvaluator::VelInType>::ConstPtr &velocity,
                                        const Evaluable<DMIErrorEvaluator::ScaleInType>::ConstPtr &scale,
                                        const DMIErrorEvaluator::DMIInType &dmi_meas);

    }  // namespace imu
}  // namespace finalicp
