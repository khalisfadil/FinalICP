#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace se3 {
        class ComposeEvaluator : public Evaluable<math::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<ComposeEvaluator>;
                using ConstPtr = std::shared_ptr<const ComposeEvaluator>;

                using InType = math::se3::Transformation;
                using OutType = math::se3::Transformation;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform1,
                        const Evaluable<InType>::ConstPtr& transform2);

                //Constructor.
                ComposeEvaluator(const Evaluable<InType>::ConstPtr& transform1,
                   const Evaluable<InType>::ConstPtr& transform2);
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
                const Evaluable<InType>::ConstPtr transform1_;              //First transformation (SE(3))
                const Evaluable<InType>::ConstPtr transform2_;              //Second transformation (SE(3))

        };

        // clang-format off
        ComposeEvaluator::Ptr compose(const Evaluable<ComposeEvaluator::InType>::ConstPtr& transform1,
                                        const Evaluable<ComposeEvaluator::InType>::ConstPtr& transform2);


    }  // namespace se3
}  // namespace finalicp