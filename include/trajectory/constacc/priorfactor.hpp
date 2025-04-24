#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/constacc/variable.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

        class PriorFactor : public Evaluable<Eigen::Matrix<double, 18, 1>> {
            public:
                using Ptr = std::shared_ptr<PriorFactor>;
                using ConstPtr = std::shared_ptr<const PriorFactor>;

                using InPoseType = math::se3::Transformation;
                using InVelType = Eigen::Matrix<double, 6, 1>;
                using InAccType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 18, 1>;

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

            protected:

                const Variable::ConstPtr knot1_;                                                    //First (earlier) knot in the trajectory
                const Variable::ConstPtr knot2_;                                                    //Second (later) knot in the trajectory.

                Eigen::Matrix<double, 18, 18> Phi_ = Eigen::Matrix<double, 18, 18>::Identity();     //Transition matrix 
                Eigen::Matrix<double, 18, 18> getJacKnot1_() const;                                 //Computes the Jacobian with respect to the first knot.
                Eigen::Matrix<double, 18, 18> getJacKnot2_() const;                                 //Computes the Jacobian with respect to the second knot.
            };
        }  // namespace const_acc
    }  // namespace traj
}  // namespace finalicp