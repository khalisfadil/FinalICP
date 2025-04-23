#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>


namespace finalicp {
    namespace vspace {
        template <int ROW = Eigen::Dynamic, int COL = ROW>
        class MatrixMultEvaluator : public Evaluable<Eigen::Matrix<double, ROW, 1>> {
            public:
                using Ptr = std::shared_ptr<MatrixMultEvaluator>;
                using ConstPtr = std::shared_ptr<const MatrixMultEvaluator>;

                using MatType = Eigen::Matrix<double, ROW, COL>;
                using InType = Eigen::Matrix<double, COL, 1>;
                using OutType = Eigen::Matrix<double, ROW, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v,
                        const MatType& s);

                //Constructor.
                MatrixMultEvaluator(const typename Evaluable<InType>::ConstPtr& v,
                        const MatType& s);

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
                const typename Evaluable<InType>::ConstPtr v_;          //Input function.
                const MatType s_;                                       //Matrix used for multiplication (reference to avoid copies).
            };

        // clang-format off
        template <int ROW, int COL = ROW>
        typename MatrixMultEvaluator<ROW, COL>::Ptr mmult(
            const typename Evaluable<typename MatrixMultEvaluator<ROW, COL>::InType>::ConstPtr& v,
            const typename MatrixMultEvaluator<ROW, COL>::MatType& s);


    }  // namespace vspace
}  // namespace finalicp

namespace finalicp {
    namespace vspace {

    template <int ROW, int COL>
    auto MatrixMultEvaluator<ROW, COL>::MakeShared(
        const typename Evaluable<InType>::ConstPtr& v, const MatType& s) -> Ptr {
        return std::make_shared<MatrixMultEvaluator>(v, s);
    }

    template <int ROW, int COL>
    MatrixMultEvaluator<ROW, COL>::MatrixMultEvaluator(
        const typename Evaluable<InType>::ConstPtr& v, const MatType& s)
        : v_(v), s_(s) {}

    template <int ROW, int COL>
    bool MatrixMultEvaluator<ROW, COL>::active() const {
        return v_->active();
    }

    template <int ROW, int COL>
    void MatrixMultEvaluator<ROW, COL>::getRelatedVarKeys(KeySet& keys) const {
        v_->getRelatedVarKeys(keys);
    }

    template <int ROW, int COL>
    auto MatrixMultEvaluator<ROW, COL>::value() const -> OutType {
        return s_ * v_->value();
    }

    template <int ROW, int COL>
    auto MatrixMultEvaluator<ROW, COL>::forward() const ->
        typename Node<OutType>::Ptr {
        const auto child = v_->forward();
        const auto value = s_ * child->value();
        const auto node = Node<OutType>::MakeShared(value);
        node->addChild(child);
        return node;
    }

    template <int ROW, int COL>
    void MatrixMultEvaluator<ROW, COL>::backward(
            const Eigen::MatrixXd& lhs, const typename Node<OutType>::Ptr& node,
            Jacobians& jacs) const {
        const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
        if (v_->active()) {
            v_->backward(lhs * s_, child, jacs);
        }
    }

    template <int ROW, int COL>
    typename MatrixMultEvaluator<ROW, COL>::Ptr mmult(
        const typename Evaluable<typename MatrixMultEvaluator<ROW, COL>::InType>::ConstPtr& v,
        const typename MatrixMultEvaluator<ROW, COL>::MatType& s) {
        return MatrixMultEvaluator<ROW, COL>::MakeShared(v, s);
    }

    }  // namespace vspace
}  // namespace finalicp