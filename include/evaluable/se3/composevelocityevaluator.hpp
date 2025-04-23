#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace se3 {
        class ComposeVelocityEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<ComposeVelocityEvaluator>;
                using ConstPtr = std::shared_ptr<const ComposeVelocityEvaluator>;

                using PoseInType = math::se3::Transformation;
                using VelInType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr& transform, const Evaluable<VelInType>::ConstPtr& velocity);

                //Constructor.
                ComposeVelocityEvaluator(const Evaluable<PoseInType>::ConstPtr& transform, const Evaluable<VelInType>::ConstPtr& velocity);

                //Checks if the acceleration error is influenced by active state variables.
                bool active() const override;

                //Collects state variable keys that influence this evaluator.
                void getRelatedVarKeys(KeySet &keys) const override;

                //Computes the acceleration error.
                OutType value() const override;

                //Forward evaluation of acceleration error.
                Node<OutType>::Ptr forward() const override;

                //Computes Jacobians for the acceleration error.
                void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const override;

            private:
                const Evaluable<PoseInType>::ConstPtr transform_;           //SE(3) transformation \( T \).
                const Evaluable<VelInType>::ConstPtr velocity_;             //6D velocity \( \xi \).

        };

        // clang-format off
        ComposeVelocityEvaluator::Ptr compose_velocity(const Evaluable<ComposeVelocityEvaluator::PoseInType>::ConstPtr& transform,
                                                        const Evaluable<ComposeVelocityEvaluator::VelInType>::ConstPtr& velocity);

    }  // namespace se3
}  // namespace finalicp