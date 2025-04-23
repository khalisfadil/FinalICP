#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>
#include <trajectory/constacc/variable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

       class AccelerationExtrapolator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<AccelerationExtrapolator>;
                using ConstPtr = std::shared_ptr<const AccelerationExtrapolator>;

                using InAccType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot);

                //Constructs an `AccelerationExtrapolator` instance.
                 AccelerationExtrapolator(const Time time, const Variable::ConstPtr& knot);

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

        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp