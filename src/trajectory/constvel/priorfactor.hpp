#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/constvel/variable.hpp>

namespace finalicp {
    namespace traj {
        namespace const_vel {

        class PriorFactor : public Evaluable<Eigen::Matrix<double, 12, 1>> {
            public:
                using Ptr = std::shared_ptr<PriorFactor>;
                using ConstPtr = std::shared_ptr<const PriorFactor>;

                using InPoseType = math::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 12, 1>;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

                //Constructs an `AccelerationExtrapolator` instance.
                PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

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

                const Variable::ConstPtr knot1_;                                                    //First (earlier) knot in the trajectory
                const Variable::ConstPtr knot2_;                                                    //Second (later) knot in the trajectory.
            };
        }  // namespace const_vel
    }  // namespace traj
}  // namespace finalicp