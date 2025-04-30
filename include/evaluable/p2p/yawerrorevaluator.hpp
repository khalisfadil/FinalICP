#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>

namespace finalicp {
    namespace p2p {
        class YawErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
            public:
                using Ptr = std::shared_ptr<YawErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const YawErrorEvaluator>;

                using PoseInType = math::se3::Transformation;
                using OutType = Eigen::Matrix<double, 1, 1>;


                //Factory method to create an instance.
                static Ptr MakeShared(const double yaw_meas, const Evaluable<PoseInType>::ConstPtr &T_ms_prev, const Evaluable<PoseInType>::ConstPtr &T_ms_curr);

                //Constructor.
                YawErrorEvaluator(const double yaw_meas, const Evaluable<PoseInType>::ConstPtr &T_ms_prev, const Evaluable<PoseInType>::ConstPtr &T_ms_curr);

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
                
            private:
                const double yaw_meas_;
                const Evaluable<PoseInType>::ConstPtr T_ms_prev_;
                const Evaluable<PoseInType>::ConstPtr T_ms_curr_;
                Eigen::Matrix<double, 1, 3> d_;
        };

        //Factory function for creating a GyroErrorEvaluator.
        YawErrorEvaluator::Ptr yawError(const double yaw_meas, const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_prev,
                                        const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_curr);
     }  // namespace p2p
}  // namespace finalicp
