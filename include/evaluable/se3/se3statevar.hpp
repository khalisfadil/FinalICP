#pragma once

#include <liegroupmath.hpp>

#include <evaluable/statevar.hpp>

namespace finalicp {
    namespace se3 {
        class SE3StateVar : public StateVar<math::se3::Transformation> {
            public:
                using Ptr = std::shared_ptr<SE3StateVar>;
                using ConstPtr = std::shared_ptr<const SE3StateVar>;

                using T = math::se3::Transformation;
                using Base = StateVar<T>;

                //Factory method to create an instance.
                static Ptr MakeShared(const T& value, const std::string& name = "");

                //Constructor.
                SE3StateVar(const T& value, const std::string& name = "");

                //Updates the velocity state using a perturbation.
                bool update(const Eigen::VectorXd& perturbation) override;

                //Collects state variable keys that influence this evaluator.
                StateVarBase::Ptr clone() const override;
        };
    }  // namespace se3
}  // namespace finalicp
