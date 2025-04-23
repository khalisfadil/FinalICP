#pragma once

#include <memory>
#include <vector>

namespace finalicp{

    //A base class representing a generic tree node with thread-safe access.
    class NodeBase {
        public:
            using Ptr = std::shared_ptr<NodeBase>;
            using ConstPtr = std::shared_ptr<const NodeBase>;

            virtual ~NodeBase() = default;

            //Adds a child node (order of addition is preserved)
            void addChild(const Ptr& child) { children_.emplace_back(child); }

            //Returns child at index
            Ptr at(const size_t& index) const { return children_.at(index); }

        private:
            std::vector<Ptr> children_;
    };

    //A templated node class that extends NodeBase to store typed values.
    template <class T>
    class Node : public NodeBase {
        public:
            using Ptr = std::shared_ptr<Node<T>>;
            using ConstPtr = std::shared_ptr<const Node<T>>;

            static Ptr MakeShared(const T& value) {
                return std::make_shared<Node<T>>(value);
            }

            Node(const T& value) : value_(value) {}

            const T& value() const { return value_; }

        private:

            T value_;

    };


}   // namespace finalicp
