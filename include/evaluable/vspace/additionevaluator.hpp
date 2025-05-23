#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace vspace {
        template <int DIM = Eigen::Dynamic>
        class AdditionEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
            public:
                using Ptr = std::shared_ptr<AdditionEvaluator>;
                using ConstPtr = std::shared_ptr<const AdditionEvaluator>;

                using InType = Eigen::Matrix<double, DIM, 1>;
                using OutType = Eigen::Matrix<double, DIM, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v1,
                        const typename Evaluable<InType>::ConstPtr& v2);

                //Constructor.
                AdditionEvaluator(const typename Evaluable<InType>::ConstPtr& v1,
                    const typename Evaluable<InType>::ConstPtr& v2);

                //Checks if the acceleration error is influenced by active state variables.
                bool active() const override;

                using KeySet = typename Evaluable<OutType>::KeySet;

                //Collects state variable keys that influence this evaluator.
                void getRelatedVarKeys(KeySet &keys) const override;

                //Computes the acceleration error.
                OutType value() const override;

                //Forward evaluation of acceleration error.
                typename Node<OutType>::Ptr forward() const override;

                //Computes Jacobians for the acceleration error.
                void backward(const Eigen::MatrixXd& lhs,const typename Node<OutType>::Ptr& node, Jacobians& jacs) const override;

            private:
                const typename Evaluable<InType>::ConstPtr v1_;         //First input function.
                const typename Evaluable<InType>::ConstPtr v2_;         //Second input function.

        };

        // clang-format off
        template <int DIM>
        typename AdditionEvaluator<DIM>::Ptr add(const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v1,
                                                    const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v2);


    }  // namespace vspace
}  // namespace finalicp

namespace finalicp {
    namespace vspace {

    template <int DIM>
    auto AdditionEvaluator<DIM>::MakeShared(
        const typename Evaluable<InType>::ConstPtr& v1,
        const typename Evaluable<InType>::ConstPtr& v2) -> Ptr {
        return std::make_shared<AdditionEvaluator>(v1, v2);
    }

    template <int DIM>
    AdditionEvaluator<DIM>::AdditionEvaluator(
        const typename Evaluable<InType>::ConstPtr& v1,
        const typename Evaluable<InType>::ConstPtr& v2)
        : v1_(v1), v2_(v2) {}

    template <int DIM>
    bool AdditionEvaluator<DIM>::active() const {
        return v1_->active() || v2_->active();
    }

    template <int DIM>
    void AdditionEvaluator<DIM>::getRelatedVarKeys(KeySet& keys) const {
        v1_->getRelatedVarKeys(keys);
        v2_->getRelatedVarKeys(keys);
    }

    template <int DIM>
    auto AdditionEvaluator<DIM>::value() const -> OutType {
        return v1_->value() + v2_->value();
    }

    template <int DIM>
    auto AdditionEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
        const auto child1 = v1_->forward();
        const auto child2 = v2_->forward();
        const auto value = child1->value() + child2->value();
        const auto node = Node<OutType>::MakeShared(value);
        node->addChild(child1);
        node->addChild(child2);
        return node;
    }

    template <int DIM>
    void AdditionEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                        const typename Node<OutType>::Ptr& node,
                                        Jacobians& jacs) const {
        const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
        const auto child2 = std::static_pointer_cast<Node<InType>>(node->at(1));

        if (v1_->active()) {
            v1_->backward(lhs, child1, jacs);
        }

        if (v2_->active()) {
            v2_->backward(lhs, child2, jacs);
        }
    }

    template <int DIM>
    typename AdditionEvaluator<DIM>::Ptr add(
        const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v1,
        const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v2) {
        return AdditionEvaluator<DIM>::MakeShared(v1, v2);
    }
    

    }  // namespace vspace
}  // namespace finalicp