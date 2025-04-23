#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/evaluable.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

        class Variable {
            public:
                using Ptr = std::shared_ptr<Variable>;
                using ConstPtr = std::shared_ptr<const Variable>;

                using PoseType = math::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;
                using AccelerationType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time time, const Evaluable<PoseType>::Ptr& T_k0,
                                        const Evaluable<VelocityType>::Ptr& w_0k_ink,
                                        const Evaluable<AccelerationType>::Ptr& dw_0k_ink) {
                    return std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink);
                }

                //Constructs a Variable.
                Variable(const Time time, const Evaluable<PoseType>::Ptr& T_k0,
                    const Evaluable<VelocityType>::Ptr& w_0k_ink,
                    const Evaluable<AccelerationType>::Ptr& dw_0k_ink)
                : time_(time), T_k0_(T_k0), w_0k_ink_(w_0k_ink), dw_0k_ink_(dw_0k_ink) {}

                //Default destructor
                virtual ~Variable() = default;

                //Get the timestamp of this variable.
                const Time& time() const { return time_; }

                //Retrieves the pose evaluable (SE(3) transformation).
                const Evaluable<PoseType>::Ptr& pose() const { return T_k0_; }

                //Retrieves the velocity evaluable (se(3) twist).
                const Evaluable<VelocityType>::Ptr& velocity() const { return w_0k_ink_; }

                //Retrieves the acceleration evaluable (se(3) acceleration).
                const Evaluable<AccelerationType>::Ptr& acceleration() const { return dw_0k_ink_; }

            private:

                Time time_;                                             //Timestamp associated with this state.
                const Evaluable<PoseType>::Ptr T_k0_;                   //Pose evaluable.
                const Evaluable<VelocityType>::Ptr w_0k_ink_;           //Velocity evaluable.
                const Evaluable<AccelerationType>::Ptr dw_0k_ink_;      //Acceleration evaluable.
            };

        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp