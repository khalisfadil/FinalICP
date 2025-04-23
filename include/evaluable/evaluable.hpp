#pragma once

#include <unordered_set>

#include <Eigen/Core>

#include <evaluable/jacobians.hpp>
#include <evaluable/node.hpp>

namespace finalicp{
    
    //Abstract base for an evaluable function with automatic differentiation support.
    template <class T>
    class Evaluable {
        public:
            using Ptr = std::shared_ptr<Evaluable<T>>;
            using ConstPtr = std::shared_ptr<const Evaluable<T>>;

            virtual ~Evaluable() = default;

            //Evaluates the function and ensures robust error handling.
            T evaluate() const { return this->value(); }


            //Evaluates the function and accumulates its Jacobians.
            T evaluate(const Eigen::MatrixXd& lhs, Jacobians& jacs) const {
                const auto end_node = this->forward();
                backward(lhs, end_node, jacs);
                return end_node->value();
            }

            //Checks if the function depends on active state variables.
            virtual bool active() const = 0;

            using KeySet = std::unordered_set<StateKey, StateKeyHash>;

            //Gathers the keys of state variables that affect this function.
            virtual void getRelatedVarKeys(KeySet& keys) const = 0;

            //Computes the value of this function (pure virtual).
            virtual T value() const = 0;

            //Forward pass: builds a Node<T> to hold the computed value for use in backward differentiation.
            virtual typename Node<T>::Ptr forward() const = 0;

            //Backward pass: accumulates Jacobians in jacs using the given lhs and the Node returned by forward().
            virtual void backward(const Eigen::MatrixXd& lhs,
                        const typename Node<T>::Ptr& node,
                        Jacobians& jacs) const = 0;
    };

}   // namespace finalicp