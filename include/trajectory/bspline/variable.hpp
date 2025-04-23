#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>
#include <evaluable/vspace/evaluables.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace bspline {

        class Variable {
            public:
                using Ptr = std::shared_ptr<Variable>;
                using ConstPtr = std::shared_ptr<const Variable>;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Time& time, const vspace::VSpaceStateVar<6>::Ptr& c) {
                    return std::make_shared<Variable>(time, c);
                }

                //Constructs a Variable.
                Variable(const Time& time, const vspace::VSpaceStateVar<6>::Ptr& c)
                    : time_(time), c_(c) {}

                //Default destructor
                virtual ~Variable() = default;

                //Get the timestamp of this variable.
                const Time& getTime() const { return time_; }

                //Get the control vector associated with this variable.
                const vspace::VSpaceStateVar<6>::Ptr& getC() const { return c_; }

            private:
            
                Time time_;                                     //Timestamp of the control point
                const vspace::VSpaceStateVar<6>::Ptr c_;        //Control vector representing the state
            };

        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp