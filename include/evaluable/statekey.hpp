#pragma once

#include <atomic>
#include <functional>

namespace finalicp{

    // -----------------------------------------------------------------------------
    /**
     * @file stateKey.hpp
     * @brief Provides a unique key system for identifying states in SLAM with robust hashing.
     */
    
    // -----------------------------------------------------------------------------
    /**
     * @typedef StateKey
     * @brief Represents a unique state key as an unsigned integer.
     */
    using StateKey = unsigned int;

    // -----------------------------------------------------------------------------
    /**
     * @struct StateKeyHash
     * @brief Robust hash function for StateKey using FNV-1a algorithm.
     */
    struct StateKeyHash {
        size_t operator()(const StateKey& key) const noexcept {
            // FNV-1a hash algorithm for better distribution
            size_t hash = 0xCBF29CE484222325ULL; // FNV-64 offset basis
            constexpr size_t prime = 0x100000001B3ULL; // FNV-64 prime

            // Mix the key
            hash ^= static_cast<size_t>(key);
            hash *= prime;

            // Additional mixing for better avalanche effect
            hash ^= hash >> 33;
            hash *= 0xFF51AFD7ED558CCDULL;
            hash ^= hash >> 33;
            hash *= 0xC4CEB9FE1A85EC53ULL;
            hash ^= hash >> 33;

            return hash;
        }
    };

    // -----------------------------------------------------------------------------
    /**
     * @struct StateKeyEqual
     * @brief Equality comparison for StateKey.
     */
    struct StateKeyEqual {
        bool operator()(const StateKey& a, const StateKey& b) const noexcept {
            return a == b;
        }
    };

    // -----------------------------------------------------------------------------
    /**
     * @struct StateKeyHashCompare
     * @brief Combined hash and equality comparison 
     */
    struct StateKeyHashCompare {
        static size_t hash(const StateKey& key) {
            return StateKeyHash{}(key);
        }

        static bool equal(const StateKey& a, const StateKey& b) {
            return StateKeyEqual{}(a, b);
        }
    };

    // -----------------------------------------------------------------------------
    /**
     * @brief Generates a new unique state key (thread-safe).
     *
     * Uses an atomic counter in relaxed mode for maximum performance
     * since only uniqueness is required (no ordering constraints).
     */
    inline StateKey NewStateKey() {
        static std::atomic<unsigned int> id{0};
        return id.fetch_add(1, std::memory_order_relaxed);
    }
}   // finalicp