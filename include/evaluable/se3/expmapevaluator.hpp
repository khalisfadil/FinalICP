#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace se3 {
        class ExpMapEvaluator : public Evaluable<math::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<ExpMapEvaluator>;
                using ConstPtr = std::shared_ptr<const ExpMapEvaluator>;

                using InType = Eigen::Matrix<double, 6, 1>;
                using OutType = math::se3::Transformation;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<InType>::ConstPtr& xi);

                //Constructor.
                ExpMapEvaluator(const Evaluable<InType>::ConstPtr& xi);

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
                const Evaluable<InType>::ConstPtr xi_;              //Twist vector \( \xi \).

        };

        // clang-format off
        ExpMapEvaluator::Ptr vec2tran(const Evaluable<ExpMapEvaluator::InType>::ConstPtr& xi);

    }  // namespace se3
}  // namespace finalicp