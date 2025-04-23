#pragma once

#include <memory>

#include <problem/optimizationproblem.hpp>

namespace finalicp{
    namespace traj {

        class Interface {
            public:
            /// Shared pointer typedefs for readability
            using Ptr = std::shared_ptr<Interface>;
            using ConstPtr = std::shared_ptr<const Interface>;

            virtual ~Interface() = default;
        };

    }  // namespace traj
}  // namespace finalicp