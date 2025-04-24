#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/constacc/variable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

        class PoseInterpolator : public Evaluable<math::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<PoseInterpolator>;
                using ConstPtr = std::shared_ptr<const PoseInterpolator>;

                using InPoseType = math::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using InAccType = Eigen::Matrix<double, 6, 1>;
                using OutType = math::se3::Transformation;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

                //Constructs an `AccelerationExtrapolator` instance.
                PoseInterpolator(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

                //Checks if the extrapolator depends on any active variables
                bool active() const override;

                //Retrieves related variable keys for factor graph optimization
                void getRelatedVarKeys(KeySet& keys) const override;

                //Computes the extrapolated acceleration value.
                OutType value() const override;

                //Computes the forward extrapolation and returns it as a node
                Node<OutType>::Ptr forward() const override;

                //Computes the backward pass, accumulating Jacobians.
                void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const override;

            protected:

                const Variable::ConstPtr knot1_;                //First (earlier) knot in the trajectory
                const Variable::ConstPtr knot2_;                //Second (later) knot in the trajectory.

                Eigen::Matrix<double, 18, 18> omega_ = Eigen::Matrix<double, 18, 18>::Zero();          //Omega matrix used for interpolation calculations
                Eigen::Matrix<double, 18, 18> lambda_ = Eigen::Matrix<double, 18, 18>::Zero();          //Lambda matrix used for interpolation calculations
            };
        }  // namespace const_acc
    }  // namespace traj
}  // namespace finalicp