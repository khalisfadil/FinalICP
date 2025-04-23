#pragma once

#include <Eigen/Core>

#include <evaluable/evaluable.hpp>

namespace finalicp {
    namespace vspace {
        template <int DIM = Eigen::Dynamic>
        class ScalarMultEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
            public:
                using Ptr = std::shared_ptr<ScalarMultEvaluator>;
                using ConstPtr = std::shared_ptr<const ScalarMultEvaluator>;

                using InType = Eigen::Matrix<double, DIM, 1>;
                using OutType = Eigen::Matrix<double, DIM, 1>;

                //Factory method to create an instance.
                static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v, const double& s);
                //Constructor.
                ScalarMultEvaluator(const typename Evaluable<InType>::ConstPtr& v, const double& s);

                //Updates the velocity state using a perturbation.
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
                const typename Evaluable<InType>::ConstPtr v_;      //Input vector function.
                const double s_;                                    //Scalar multiplier.
        };

        template <int DIM>
        typename ScalarMultEvaluator<DIM>::Ptr smult(
            const typename Evaluable<typename ScalarMultEvaluator<DIM>::InType>::ConstPtr& v,
            const double& s);

    }  // namespace vspace
}  // namespace finalicp

namespace finalicp {
    namespace vspace {

        template <int DIM>
        auto ScalarMultEvaluator<DIM>::MakeShared(
            const typename Evaluable<InType>::ConstPtr& v, const double& s) -> Ptr {
            return std::make_shared<ScalarMultEvaluator>(v, s);
        }

        template <int DIM>
        ScalarMultEvaluator<DIM>::ScalarMultEvaluator(
            const typename Evaluable<InType>::ConstPtr& v, const double& s)
            : v_(v), s_(s) {}

        template <int DIM>
        bool ScalarMultEvaluator<DIM>::active() const {
            return v_->active();
        }

        template <int DIM>
        void ScalarMultEvaluator<DIM>::getRelatedVarKeys(KeySet& keys) const {
            v_->getRelatedVarKeys(keys);
        }

        template <int DIM>
        auto ScalarMultEvaluator<DIM>::value() const -> OutType {
            return s_ * v_->value();
        }

        template <int DIM>
        auto ScalarMultEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
            const auto child = v_->forward();
            const auto value = s_ * child->value();
            const auto node = Node<OutType>::MakeShared(value);
            node->addChild(child);
            return node;
        }

        template <int DIM>
        void ScalarMultEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                                const typename Node<OutType>::Ptr& node,
                                                Jacobians& jacs) const {
            const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
            if (v_->active()) {
                v_->backward(s_ * lhs, child, jacs);
            }
        }

        template <int DIM>
        typename ScalarMultEvaluator<DIM>::Ptr smult(
            const typename Evaluable<typename ScalarMultEvaluator<DIM>::InType>::ConstPtr& v,
            const double& s) {
            return ScalarMultEvaluator<DIM>::MakeShared(v, s);
        }
    

    }  // namespace vspace
}  // namespace finalicp
