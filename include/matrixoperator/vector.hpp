#pragma once

#include <vector>
#include <map>

#include <Eigen/Core>

#include <matrixoperator/matrixindexing.hpp>

namespace finalicp {
    class BlockVector {
    public:
        // Default constructor
        BlockVector();

        // Block size constructor
        BlockVector(const std::vector<unsigned int>& blkRowSizes);

        // Block size (with data) constructor
        BlockVector(const std::vector<unsigned int>& blkRowSizes, const Eigen::VectorXd& data);

        // Set internal data (total size of v must match concatenated block sizes)
        void setFromScalar(const Eigen::VectorXd& v);

        // Get indexing object
        const BlockDimIndexing& getIndexing() const;

        // Adds the vector to the block entry at index 'r', block dim must match
        void add(unsigned int r, const Eigen::VectorXd& v);

        // Return block vector at index 'r'
        Eigen::VectorXd at(unsigned int r) const;

        // Return map to block vector at index 'r' (read-only)
        Eigen::Map<const Eigen::VectorXd> mapAt(unsigned int r) const;

        // Return map to block vector at index 'r' (writable)
        Eigen::Map<Eigen::VectorXd> mapAt(unsigned int r);

        // Convert to Eigen dense vector format
        const Eigen::VectorXd& toEigen() const; // Added const

    private:
        // Block indexing object
        BlockDimIndexing indexing_;

        // Vector data
        Eigen::VectorXd data_;
    };
} // namespace finalicp