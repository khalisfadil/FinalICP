#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>

namespace finalicp {
    namespace p2p {
        class YawVelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
            public:
                using Ptr = std::shared_ptr<YawVelErrorEvaluator>;
                using ConstPtr = std::shared_ptr<const YawVelErrorEvaluator>;

                using InType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 1, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(Eigen::Matrix<double, 1, 1> vel_meas, const Evaluable<InType>::ConstPtr &w_iv_inv);

                //Constructor.
                YawVelErrorEvaluator(Eigen::Matrix<double, 1, 1> vel_meas, const Evaluable<InType>::ConstPtr &w_iv_inv);

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
                const Eigen::Matrix<double, 1, 1> vel_meas_;
                const Evaluable<InType>::ConstPtr w_iv_inv_;
                Eigen::Matrix<double, 1, 6> D_; // pick out yaw vel
        };

        //Factory function for creating a GyroErrorEvaluator.
        YawVelErrorEvaluator::Ptr velError(const Eigen::Matrix<double, 1, 1> vel_meas,
                                            const Evaluable<YawVelErrorEvaluator::InType>::ConstPtr &w_iv_inv);
     }  // namespace p2p
}  // namespace finalicp
