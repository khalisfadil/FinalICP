#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace se3 {
        class SE3ErrorGlobalPerturbEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<SE3ErrorGlobalPerturbEvaluator>;
                using ConstPtr = std::shared_ptr<const SE3ErrorGlobalPerturbEvaluator>;

                using InType = math::se3::Transformation;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_ab, const InType& T_ab_meas);

                //Constructor.
                SE3ErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr& T_ab, const InType& T_ab_meas);

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
                const Evaluable<InType>::ConstPtr T_ab_;            //Estimated SE(3) transformation \( T_{ab} \).
                const InType T_ab_meas_;                            //Measured SE(3) transformation \( T_{ab}^{meas} \).

        };

        // clang-format off
        SE3ErrorGlobalPerturbEvaluator::Ptr se3_global_perturb_error(const Evaluable<SE3ErrorGlobalPerturbEvaluator::InType>::ConstPtr& T_ab,
                                                                        const SE3ErrorGlobalPerturbEvaluator::InType& T_ab_meas);
    }  // namespace se3
}  // namespace finalicp