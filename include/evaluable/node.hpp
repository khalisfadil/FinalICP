#pragma once

#include <memory>
#include <vector>
#include <iostream>

namespace finalicp{

    //A base class representing a generic tree node with thread-safe access.
    class NodeBase {
        public:
            using Ptr = std::shared_ptr<NodeBase>;
            using ConstPtr = std::shared_ptr<const NodeBase>;

            virtual ~NodeBase() = default;

            //Adds a child node (order of addition is preserved)
            void addChild(const Ptr& child) { 
                //debug
                // ################################
                std::cout << "[DEBUG::node] Adding child to NodeBase: " << this << ", child: " << child.get() << ", use_count: " << child.use_count() << std::endl;
                 // ################################
                children_.emplace_back(child); }

            //Returns child at index
            Ptr at(const size_t& index) const { 
                //debug
                // ################################
                std::cout << "[DEBUG::node] Accessing child at index: " << index << " from NodeBase: " << this << std::endl;
                 // ################################
                return children_.at(index); }

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
                // Debug
                // ###################
                // std::cout << "[DEBUG::node] Creating Node<" << typeid(T).name() << "> with value" << std::endl;
                // ###################

                return std::make_shared<Node<T>>(value);
            }

            Node(const T& value) : value_(value) {
                // Debug
                // ###################
                // std::cout << "[DEBUG::node] Constructing Node<" << typeid(T).name() << ">: " << this << std::endl;
                // ###################
            }

            const T& value() const { return value_; }

        private:

            T value_;

    };


}   // namespace finalicp
