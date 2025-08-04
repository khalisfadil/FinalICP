#include <trajectory/constacc/evaluable/composecurlyhatevaluator.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc{

            // ###########################################################
            // MakeShared
            // ###########################################################

            auto ComposeCurlyhatEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& x, const Evaluable<InType>::ConstPtr& y)-> Ptr {
                return std::make_shared<ComposeCurlyhatEvaluator>(x, y);
            }

            // ###########################################################
            // ComposeCurlyhatEvaluator
            // ###########################################################

            ComposeCurlyhatEvaluator::ComposeCurlyhatEvaluator(const Evaluable<InType>::ConstPtr& x, const Evaluable<InType>::ConstPtr& y)
                : x_(x), y_(y) {}

            // ###########################################################
            // active
            // ###########################################################

            bool ComposeCurlyhatEvaluator::active() const {
                return x_->active() || y_->active();
            }

            // ###########################################################
            // getRelatedVarKeys
            // ###########################################################

            void ComposeCurlyhatEvaluator::getRelatedVarKeys(KeySet& keys) const {
                x_->getRelatedVarKeys(keys);
                y_->getRelatedVarKeys(keys);
            }

            // ###########################################################
            // value
            // ###########################################################

            auto ComposeCurlyhatEvaluator::value() const -> OutType {
                const auto x_val = x_->value();
                const auto y_val = y_->value();
                OutType result = math::se3::curlyhat(x_val) * y_val;

#ifdef DEBUG
                // --- [IMPROVEMENT] Check for numerical instability in the forward pass ---
                if (!x_val.allFinite() || !y_val.allFinite()) {
                    std::cerr << "[CONSTACC ComposeCurlyhat DEBUG | value] CRITICAL: Input values are non-finite!" << std::endl;
                }
                if (!result.allFinite()) {
                    std::cerr << "[CONSTACC ComposeCurlyhat DEBUG | value] CRITICAL: Output value is non-finite!" << std::endl;
                }
#endif
                return result;
            }

            // ###########################################################
            // forward
            // ###########################################################

            auto ComposeCurlyhatEvaluator::forward() const -> Node<OutType>::Ptr {
                const auto x = x_->forward();
                const auto y = y_->forward();
                const auto value = math::se3::curlyhat(x->value()) * y->value();

#ifdef DEBUG
                if (!value.allFinite()) {
                    std::cerr << "[CONSTACC ComposeCurlyhat DEBUG | forward] CRITICAL: Node value is non-finite!" << std::endl;
                }
#endif

                const auto node = Node<OutType>::MakeShared(value);
                node->addChild(x);
                node->addChild(y);
                return node;
            }

            // ###########################################################
            // backward
            // ###########################################################

            void ComposeCurlyhatEvaluator::backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node, Jacobians& jacs) const {
                const auto x_node = std::static_pointer_cast<Node<InType>>(node->at(0));
                const auto y_node = std::static_pointer_cast<Node<InType>>(node->at(1));

#ifdef DEBUG
                // --- [IMPROVEMENT] Check the incoming derivative and outgoing derivatives ---
                if (!lhs.allFinite()) {
                    std::cerr << "[CONSTACC ComposeCurlyhat DEBUG | backward] CRITICAL: Incoming derivative (lhs) is non-finite!" << std::endl;
                }
#endif

                if (x_->active()) {
                    // Derivative w.r.t. x is: -curlyhat(y)
                    const Eigen::MatrixXd new_lhs_x = (-1.0) * lhs * math::se3::curlyhat(y_node->value());
#ifdef DEBUG
                    if (!new_lhs_x.allFinite()) {
                        std::cerr << "[CONSTACC ComposeCurlyhat DEBUG | backward] CRITICAL: Derivative propagated to child 'x' is non-finite!" << std::endl;
                    }
#endif
                    x_->backward(new_lhs_x, x_node, jacs);
                }

                if (y_->active()) {
                    // Derivative w.r.t. y is: curlyhat(x)
                    const Eigen::MatrixXd new_lhs_y = lhs * math::se3::curlyhat(x_node->value());
#ifdef DEBUG
                    if (!new_lhs_y.allFinite()) {
                        std::cerr << "[CONSTACC ComposeCurlyhat DEBUG | backward] CRITICAL: Derivative propagated to child 'y' is non-finite!" << std::endl;
                    }
#endif
                    y_->backward(new_lhs_y, y_node, jacs);
                }
            }

            // ###########################################################
            // compose_curlyhat
            // ###########################################################

            ComposeCurlyhatEvaluator::Ptr compose_curlyhat(const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& x, const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& y) {
                return ComposeCurlyhatEvaluator::MakeShared(x, y);
            }
            
        } // namespace const_acc
    }  // namespace traj
}  // namespace finalicp