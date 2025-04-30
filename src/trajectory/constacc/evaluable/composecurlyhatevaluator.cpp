#include <trajectory/constacc/evaluable/composecurlyhatevaluator.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc{

            auto ComposeCurlyhatEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& x, const Evaluable<InType>::ConstPtr& y)-> Ptr {
                return std::make_shared<ComposeCurlyhatEvaluator>(x, y);
            }

            ComposeCurlyhatEvaluator::ComposeCurlyhatEvaluator(const Evaluable<InType>::ConstPtr& x, const Evaluable<InType>::ConstPtr& y)
                : x_(x), y_(y) {}

            bool ComposeCurlyhatEvaluator::active() const {
                return x_->active() || y_->active();
            }

            void ComposeCurlyhatEvaluator::getRelatedVarKeys(KeySet& keys) const {
                x_->getRelatedVarKeys(keys);
                y_->getRelatedVarKeys(keys);
            }

            auto ComposeCurlyhatEvaluator::value() const -> OutType {
                return math::se3::curlyhat(x_->value()) * y_->value();
            }

            auto ComposeCurlyhatEvaluator::forward() const -> Node<OutType>::Ptr {
                //
                const auto x = x_->forward();
                const auto y = y_->forward();

                //
                const auto value = math::se3::curlyhat(x->value()) * y->value();

                //
                const auto node = Node<OutType>::MakeShared(value);
                node->addChild(x);
                node->addChild(y);

                return node;
            }

            void ComposeCurlyhatEvaluator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                const auto x = std::static_pointer_cast<Node<InType>>(node->at(0));
                const auto y = std::static_pointer_cast<Node<InType>>(node->at(1));

                if (x_->active())
                    x_->backward((-1.0) * lhs * math::se3::curlyhat(y->value()), x, jacs);

                if (y_->active())
                    y_->backward(lhs * math::se3::curlyhat(x->value()), y, jacs);
            }

            ComposeCurlyhatEvaluator::Ptr compose_curlyhat(const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& x, const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& y) {
                return ComposeCurlyhatEvaluator::MakeShared(x, y);
            }
            
        } // namespace const_acc
    }  // namespace traj
}  // namespace finalicp