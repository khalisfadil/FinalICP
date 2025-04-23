#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>

#include <evaluable/statevar.hpp>

namespace finalicp {
    namespace vspace {
        template <int DIM = Eigen::Dynamic>
        class VSpaceStateVar : public StateVar<Eigen::Matrix<double, DIM, 1>> {
            public:
                using Ptr = std::shared_ptr<VSpaceStateVar>;
                using ConstPtr = std::shared_ptr<const VSpaceStateVar>;

                using T = Eigen::Matrix<double, DIM, 1>;
                using Base = StateVar<T>;

                //Factory method to create an instance.
                static Ptr MakeShared(const T& value, const std::string& name = "");

                //Constructor.
                VSpaceStateVar(const T& value, const std::string& name = "");

                //Updates the velocity state using a perturbation.
                bool update(const Eigen::VectorXd& perturbation) override;

                using KeySet = typename Evaluable<OutType>::KeySet;

                //Collects state variable keys that influence this evaluator.
                StateVarBase::Ptr clone() const override;

            private:
                const typename Evaluable<InType>::ConstPtr bias1_;          //First bias state.
                const typename Evaluable<InType>::ConstPtr bias2_;          //Second bias state.
                double psi_;    ///< Interpolation weight for `bias2`.
                double lambda_; ///< Interpolation weight for `bias1`.
        };

        template <int DIM>
        auto VSpaceStateVar<DIM>::MakeShared(const T& value, const std::string& name)
            -> Ptr {
            return std::make_shared<VSpaceStateVar<DIM>>(value, name);
        }

        template <int DIM>
        VSpaceStateVar<DIM>::VSpaceStateVar(const T& value, const std::string& name)
            : Base(value, DIM, name) {}

            template <int DIM>
            bool VSpaceStateVar<DIM>::update(const Eigen::VectorXd& perturbation) {
            if (perturbation.size() != this->perturb_dim())
                throw std::runtime_error(
                    "VSpaceStateVar::update: perturbation size mismatch");
            //
            this->value_ = this->value_ + perturbation;
            return true;
        }

        template <int DIM>
        StateVarBase::Ptr VSpaceStateVar<DIM>::clone() const {
            return std::make_shared<VSpaceStateVar<DIM>>(*this);
        }
    }  // namespace vspace
}  // namespace finalicp
