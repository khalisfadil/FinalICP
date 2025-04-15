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
            throw std::invalid_argument("[BlockVector] Block row size: " + std::to_string(indexing_.scalarSize()) +
                                    " and vector size: " + std::to_string(v.size()) + " do not match.");
        }
        data_ = v;
    }

    const BlockDimIndexing& BlockVector::getIndexing() {
        return indexing_;
    }

    void BlockVector::add(const unsigned int& r, const Eigen::VectorXd& v) {
        if (r >= indexing_.numEntries()) {
            throw std::invalid_argument("[BlockVector] Requested row index out of bounds");
        }
        if (v.size() != indexing_.blkSizeAt(r)) {
            throw std::invalid_argument("[BlockVector] Block row size and vector size do not match");
        }
        data_.segment(indexing_.cumSumAt(r), indexing_.blkSizeAt(r)) += v;
    }

    Eigen::VectorXd BlockVector::at(const unsigned int& r) {
        if (r >= indexing_.numEntries()) {
            throw std::invalid_argument("Requested row index out of bounds");
        }
        return data_.segment(indexing_.cumSumAt(r), indexing_.blkSizeAt(r));
    }

    Eigen::Map<Eigen::VectorXd> BlockVector::mapAt(const unsigned int& r) {
        if (r >= indexing_.numEntries()) {
            throw std::invalid_argument("Requested row index out of bounds");
        }
        return Eigen::Map<Eigen::VectorXd>(data_.data() + indexing_.cumSumAt(r), indexing_.blkSizeAt(r));
    }

    const Eigen::VectorXd& BlockVector::toEigen() {
        return data_;
    }

} // namespace finalicp