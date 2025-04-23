#pragma once

#include <Eigen/Dense>

#include <evaluable/evaluable.hpp> 
#include <evaluable/statekey.hpp> 

namespace finalicp{
    //Base class for all state variables in the optimization framework.
    class StateVarBase {
        public:

            using Ptr = std::shared_ptr<StateVarBase>;
            using ConstPtr = std::shared_ptr<const StateVarBase>;

            //Constructor for the base state variable.
            StateVarBase(const unsigned int& perturb_dim, const std::string& name = "")
                : perturb_dim_(perturb_dim), name_(name) {}

            //Returns the name of the state variable.
            std::string name() const { return name_; }

            //Updates this state variable from a perturbation.
            virtual bool update(const Eigen::VectorXd& perturbation) = 0;

            //Returns a clone of the state variable.
            virtual Ptr clone() const = 0;

            //Sets the state value from anotehr instance of the state
            virtual void setFromCopy(const ConstPtr& other) = 0;

            //eturns the unique key associated with this state variable.
            const StateKey& key() const { return key_; }

            //Returns the perturbation dimension of this state variable.
            const unsigned int& perturb_dim() const { return perturb_dim_; }

            //Checks if the state variable is locked (i.e., not updated in optimization).
            const bool& locked() const { return locked_; }

            //Allows modification of the lock status.
            bool& locked() { return locked_; }

        private:
            const unsigned int perturb_dim_;        //Dimension of the perturbation
            const std::string name_;                //Name of the state variable
            const StateKey key_ = NewStateKey();    //Unique identifier for the state
            bool locked_ = false;                   //Indicates whether the state is locked (non-optimizable)
    };

    template <class T>
    //Templated class representing a state variable of type T.
    class StateVar : public StateVarBase, public Evaluable<T> {

        public:
            using Ptr = std::shared_ptr<StateVar<T>>;
            using ConstPtr = std::shared_ptr<const StateVar<T>>;

            //Constructs a state variable with an initial value.
            StateVar(const T& value, const unsigned int& perturb_dim,
                const std::string& name = "")
            : StateVarBase(perturb_dim, name), value_(value) {}

            //Copies the value from another state variable instance.
            void setFromCopy(const StateVarBase::ConstPtr& other) override {
                if (key() != other->key())
                throw std::runtime_error("StateVar::setFromCopy: keys do not match");
                value_ = std::static_pointer_cast<const StateVar<T>>(other)->value_;
            }

            // Checks if the state variable is active in the optimization.
            bool active() const override { return !locked(); }
                using KeySet = typename Evaluable<T>::KeySet;
                void getRelatedVarKeys(KeySet& keys) const override {
                    if (!locked()) keys.insert(key());
            }

            T value() const override { return value_; }
                typename Node<T>::Ptr forward() const override {
                    return Node<T>::MakeShared(value_);
            }

            void backward(const Eigen::MatrixXd& lhs, const typename Node<T>::Ptr& node,
                            Jacobians& jacs) const override {
                if (active()) jacs.add(key(), lhs);
            }

        private: 

            T value_;

    };
}