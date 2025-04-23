#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>
#include <trajectory/constvel/variable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace const_vel {

        class PoseExtrapolator : public Evaluable<math::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<PoseExtrapolator>;
                using ConstPtr = std::shared_ptr<const PoseExtrapolator>;

                using InPoseType = math::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using OutType = math::se3::Transformation;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot);

                //Constructs an `AccelerationExtrapolator` instance.
                PoseExtrapolator(const Time time, const Variable::ConstPtr& knot);

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

            private:

                const Variable::ConstPtr knot_;                 //The knot (state) to extrapolate from.
                Eigen::Matrix<double, 18, 18> Phi_;             //Transition matrix for constant acceleration extrapolation

            };

        }  // namespace const_vel
    }  // namespace traj
}  // namespace finalicp