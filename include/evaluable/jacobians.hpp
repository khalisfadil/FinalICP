#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <evaluable/statekey.hpp> 

namespace finalicp {
    // Thread-safe container for managing and accumulating Jacobians.
    class Jacobians {
    public:
        // Concurrent Hash Map Definition
        using KeyJacMap = std::unordered_map<StateKey, Eigen::MatrixXd, StateKeyHash>;

        // Inserts or accumulates a Jacobian for a given state key.
        void add(const StateKey &key, const Eigen::MatrixXd &jac) {
            auto iter_success = jacs_.try_emplace(key, jac);
            if (!iter_success.second) iter_success.first->second += jac;
        }

        // Clears all stored Jacobians.
        void clear() { jacs_.clear(); }

        // Returns a reference to the internal concurrent hash map.
        KeyJacMap& get() { return jacs_; }

    private:
        KeyJacMap jacs_;        //Concurrent hash map storing state Jacobians.
    };

} // namespace finalicp