#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>

namespace finalicp {
    namespace p2p {
        class VelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 2, 1>> {
            public:
                using Ptr = std::shared_ptr<VelErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const VelErrorEvaluator>;

                using InType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 2, 1>;


                //Factory method to create an instance.
                static Ptr MakeShared(const Eigen::Vector2d vel_meas, const Evaluable<InType>::ConstPtr &w_iv_inv);

                //Constructor.
                VelErrorEvaluator(const Eigen::Vector2d vel_meas, const Evaluable<InType>::ConstPtr &w_iv_inv);

                //Checks if the DMI error is influenced by active state variables.
                bool active() const override;

                //Collects state variable keys that influence this evaluator.
                void getRelatedVarKeys(KeySet &keys) const override;

                //Computes the DMI error.
                OutType value() const override;

                //Forward evaluation of DMI error.
                Node<OutType>::Ptr forward() const override;

                //Computes Jacobians for the DMI error.
                void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node, Jacobians &jacs) const override;
                
            private:
                const Evaluable<InType>::ConstPtr w_iv_inv_;
                const Eigen::Vector2d vel_meas_;
                Eigen::Matrix<double, 2, 6> D_;
        };

        //Factory function for creating a GyroErrorEvaluator.
        VelErrorEvaluator::Ptr velError(const Eigen::Vector2d vel_meas,const Evaluable<VelErrorEvaluator::InType>::ConstPtr &w_iv_inv);
     }  // namespace p2p
}  // namespace finalicp
