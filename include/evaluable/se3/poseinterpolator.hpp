#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/time.hpp>


namespace finalicp {
    namespace se3 {
        class PoseInterpolator : public Evaluable<math::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<PoseInterpolator>;
                using ConstPtr = std::shared_ptr<const PoseInterpolator>;
                using Time = traj::Time;

                using InType = math::se3::Transformation;
                using OutType = math::se3::Transformation;

                //Factory method to create an instance.
                static Ptr MakeShared(const Time& time,
                                        const Evaluable<InType>::ConstPtr& transform1,
                                        const Time& time1,
                                        const Evaluable<InType>::ConstPtr& transform2,
                                        const Time& time2);

                //Constructor.
                PoseInterpolator(const Time& time,
                                const Evaluable<InType>::ConstPtr& transform1,
                                const Time& time1,
                                const Evaluable<InType>::ConstPtr& transform2,
                                const Time& time2);

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
                const Evaluable<InType>::ConstPtr transform1_;      //First transformation \(T_1\).
                const Evaluable<InType>::ConstPtr transform2_;      //Second transformation \(T_2\).
                double alpha_;                                      //Interpolation factor \( \alpha \).
                std::vector<double> faulhaber_coeffs_;              //**Precomputed Faulhaber coefficients**.

        };

        
    }  // namespace se3
}  // namespace finalicp