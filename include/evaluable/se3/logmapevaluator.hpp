#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace se3 {
        class LogMapEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<LogMapEvaluator>;
                using ConstPtr = std::shared_ptr<const LogMapEvaluator>;

                using InType = math::se3::Transformation;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform);

                //Constructor.
                LogMapEvaluator(const Evaluable<InType>::ConstPtr& transform);

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
                const Evaluable<InType>::ConstPtr transform_;           //SE(3) transformation \( T \).

        };

        // clang-format off
        LogMapEvaluator::Ptr tran2vec(const Evaluable<LogMapEvaluator::InType>::ConstPtr& transform);
    }  // namespace se3
}  // namespace finalicp