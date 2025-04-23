#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace traj {
        namespace const_acc{
        class ComposeCurlyhatEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<ComposeCurlyhatEvaluator>;
                using ConstPtr = std::shared_ptr<const ComposeCurlyhatEvaluator>;

                using InType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<InType>::ConstPtr& x,
                        const Evaluable<InType>::ConstPtr& y);

                //Constructor.
                ComposeCurlyhatEvaluator(const Evaluable<InType>::ConstPtr& x,
                           const Evaluable<InType>::ConstPtr& y);

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
                const Evaluable<InType>::ConstPtr x_;           //First input evaluator (se(3) vector).
                const Evaluable<InType>::ConstPtr y_;           //Second input evaluator (se(3) vector).

        };

        // clang-format off
        ComposeCurlyhatEvaluator::Ptr compose_curlyhat(const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& x,
                                                        const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& y);

        } // namespace const_acc
    }  // namespace traj
}  // namespace finalicp