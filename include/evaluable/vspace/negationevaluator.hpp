#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace vspace {
        template <int DIM = Eigen::Dynamic>
        class NegationEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
            public:
                using Ptr = std::shared_ptr<NegationEvaluator>;
                using ConstPtr = std::shared_ptr<const NegationEvaluator>;

                using InType = Eigen::Matrix<double, DIM, 1>;
                using OutType = Eigen::Matrix<double, DIM, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v);

                //Constructor.
                NegationEvaluator(const typename Evaluable<InType>::ConstPtr& v);

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
                typename Node<OutType>::Ptr forward() const override;       //Input function.
            };

        // clang-format off
        template <int DIM>
        typename NegationEvaluator<DIM>::Ptr neg(
            const typename Evaluable<typename NegationEvaluator<DIM>::InType>::ConstPtr& v);


    }  // namespace vspace
}  // namespace finalicp

namespace finalicp {
    namespace vspace {

    template <int DIM>
    auto NegationEvaluator<DIM>::MakeShared(
        const typename Evaluable<InType>::ConstPtr& v) -> Ptr {
        return std::make_shared<NegationEvaluator>(v);
    }

    template <int DIM>
    NegationEvaluator<DIM>::NegationEvaluator(
        const typename Evaluable<InType>::ConstPtr& v)
        : v_(v) {}

    template <int DIM>
    bool NegationEvaluator<DIM>::active() const {
        return v_->active();
    }

    template <int DIM>
    void NegationEvaluator<DIM>::getRelatedVarKeys(KeySet& keys) const {
        v_->getRelatedVarKeys(keys);
    }

    template <int DIM>
    auto NegationEvaluator<DIM>::value() const -> OutType {
        return -v_->value();
    }

    template <int DIM>
    auto NegationEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
        const auto child = v_->forward();
        const auto value = -child->value();
        const auto node = Node<OutType>::MakeShared(value);
        node->addChild(child);
        return node;
    }

    template <int DIM>
        void NegationEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                            const typename Node<OutType>::Ptr& node,
                                            Jacobians& jacs) const {
        const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
        if (v_->active()) {
            v_->backward(-lhs, child, jacs);
        }
    }

    template <int DIM>
    typename NegationEvaluator<DIM>::Ptr neg(
        const typename Evaluable<typename NegationEvaluator<DIM>::InType>::ConstPtr& v) {
        return NegationEvaluator<DIM>::MakeShared(v);
    }

    }  // namespace vspace
}  // namespace finalicp