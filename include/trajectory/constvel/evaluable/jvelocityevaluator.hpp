#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace traj {
        namespace const_vel{
        class JVelocityEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<JVelocityEvaluator>;
                using ConstPtr = std::shared_ptr<const JVelocityEvaluator>;

                using XiInType = Eigen::Matrix<double, 6, 1>;
                using VelInType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<XiInType>::ConstPtr& xi, const Evaluable<VelInType>::ConstPtr& velocity);

                //Constructor.
                JVelocityEvaluator(const Evaluable<XiInType>::ConstPtr& xi, const Evaluable<VelInType>::ConstPtr& velocity);

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
                const Evaluable<XiInType>::ConstPtr xi_;           //First input evaluator (se(3) vector).
                const Evaluable<VelInType>::ConstPtr velocity_;           //Second input evaluator (se(3) vector).

        };

        // clang-format off
        JVelocityEvaluator::Ptr j_velocity(const Evaluable<JVelocityEvaluator::XiInType>::ConstPtr& xi,
                                            const Evaluable<JVelocityEvaluator::VelInType>::ConstPtr& velocity);
        } // namespace const_vel
    }  // namespace traj
}  // namespace finalicp