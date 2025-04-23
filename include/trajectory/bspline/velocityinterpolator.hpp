#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>
#include <trajectory/bspline/variable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace bspline {

        class VelocityInterpolator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
            public:
                using Ptr = std::shared_ptr<VelocityInterpolator>;
                using ConstPtr = std::shared_ptr<const VelocityInterpolator>;

                using CType = Eigen::Matrix<double, 6, 1>;
                using OutType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create a shared instance of VelocityInterpolator.
                static Ptr MakeShared(const Time& time, const Variable::ConstPtr& k1,
                        const Variable::ConstPtr& k2,
                        const Variable::ConstPtr& k3,
                        const Variable::ConstPtr& k4);

                //Constructor for VelocityInterpolator.
                VelocityInterpolator(const Time& time, const Variable::ConstPtr& k1,
                       const Variable::ConstPtr& k2,
                       const Variable::ConstPtr& k3,
                       const Variable::ConstPtr& k4);

                //Checks if the interpolator is active (any control point is active).
                bool active() const override;

                //Retrieves keys of related variables.
                void getRelatedVarKeys(KeySet& keys) const override;

                //Computes the interpolated velocity value
                OutType value() const override;

                //Computes the forward evaluation of velocity
                Node<OutType>::Ptr forward() const override;

                //Computes the backward propagation of Jacobians
                void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const override;

            private:
            
                const Variable::ConstPtr k1_;           //Control points for interpolation
                const Variable::ConstPtr k2_;           //Control points for interpolation
                const Variable::ConstPtr k3_;           //Control points for interpolation
                const Variable::ConstPtr k4_;           //Control points for interpolation

                Eigen::Matrix<double, 4, 1> w_;         //B-spline weights for interpolation
            };
        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp