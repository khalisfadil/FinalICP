#include <matrixoperator/vector.hpp>

#include <stdexcept>

namespace finalicp{

    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------

    BlockVector::BlockVector() {}

    BlockVector::BlockVector(const std::vector<unsigned int>& blkRowSizes)
        : indexing_(blkRowSizes), data_(Eigen::VectorXd::Zero(indexing_.scalarSize())) {}

    BlockVector::BlockVector(const std::vector<unsigned int>& blkRowSizes, const Eigen::VectorXd& v)
        : indexing_(blkRowSizes), data_(v) {
        if (v.size() != indexing_.scalarSize()) {
            throw std::invalid_argument("[BlockVector] Vector size: " + std::to_string(v.size()) +
                                    " does not match block row size: " + std::to_string(indexing_.scalarSize()));
        }
    }

    // -----------------------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------------------

    void BlockVector::setFromScalar(const Eigen::VectorXd& v) {
        if (indexing_.scalarSize() != static_cast<unsigned int>(v.size())) {
            throw std::invalid_argument("[BlockVector::setFromScalar] Block row size: " + std::to_string(indexing_.scalarSize()) +
                                    " and vector size: " + std::to_string(v.size()) + " do not match.");
        }
        data_ = v;
    }

    const BlockDimIndexing& BlockVector::getIndexing() const {
        return indexing_;
    }

    void BlockVector::add(unsigned int r, const Eigen::VectorXd& v) {
        // Validate row index
        if (r >= indexing_.numEntries()) {
            throw std::invalid_argument("[BlockVector::add] Invalid row index: " + std::to_string(r));
        }

        // Cache block size
        unsigned int blkSize = indexing_.blkSizeAt(r);

        // Validate vector size
        if (v.rows() != static_cast<int>(blkSize)) {
            throw std::invalid_argument("[BlockVector::add] Vector size (" + std::to_string(v.rows()) + 
                                    ") does not match block size (" + std::to_string(blkSize) + ")");
        }

        // Add vector to the corresponding block
        data_.segment(indexing_.cumSumAt(r), blkSize) += v;
    }

    Eigen::VectorXd BlockVector::at(unsigned int r) const {
        // Validate row index
        if (r >= indexing_.numEntries()) {
            throw std::invalid_argument("[BlockVector::at] Invalid row index: " + std::to_string(r));
        }

        // Cache block size and offset
        unsigned int blkSize = indexing_.blkSizeAt(r);
        unsigned int offset = indexing_.cumSumAt(r);

        // Return copy of the block
        return data_.segment(offset, blkSize);
    }

    Eigen::Map<const Eigen::VectorXd> BlockVector::mapAt(unsigned int r) const {
        if (r >= indexing_.numEntries()) {
            throw std::invalid_argument("[BlockVector::mapAt] Invalid row index: " + std::to_string(r));
        }
        unsigned int blkSize = indexing_.blkSizeAt(r);
        unsigned int offset = indexing_.cumSumAt(r);
        return Eigen::Map<const Eigen::VectorXd>(data_.data() + offset, blkSize);
    }

    Eigen::Map<Eigen::VectorXd> BlockVector::mapAt(unsigned int r) {
        if (r >= indexing_.numEntries()) {
            throw std::invalid_argument("[BlockVector::mapAt] Invalid row index: " + std::to_string(r));
        }
        unsigned int blkSize = indexing_.blkSizeAt(r);
        unsigned int offset = indexing_.cumSumAt(r);
        return Eigen::Map<Eigen::VectorXd>(data_.data() + offset, blkSize);
    }

    const Eigen::VectorXd& BlockVector::toEigen() const {
        return data_;
    }

} // namespace finalicp