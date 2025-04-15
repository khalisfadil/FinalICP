#include <matrixoperator/matrixbase.hpp>

#include <stdexcept>
#include <iostream>

namespace finalicp {
    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------

    BlockMatrixBase::BlockMatrixBase() {}

    BlockMatrixBase::BlockMatrixBase(const std::vector<unsigned int>& blkRowSizes,
                                 const std::vector<unsigned int>& blkColSizes)
        : indexing_(blkRowSizes, blkColSizes), symmetric_(false) {
        if (blkRowSizes.empty()) {
            throw std::invalid_argument("[BlockMatrixBase] No row sizes provided.");
        }
        if (blkColSizes.empty()) {
            throw std::invalid_argument("[BlockMatrixBase] No column sizes provided.");
        }
    }

    BlockMatrixBase::BlockMatrixBase(const std::vector<unsigned int>& blkSqSizes, bool symmetric)
        : indexing_(blkSqSizes), symmetric_(symmetric) {
        if (blkSqSizes.empty()) {
            throw std::invalid_argument("[BlockMatrixBase] No block sizes provided.");
        }
    }

    // -----------------------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------------------

    const BlockMatrixIndexing& BlockMatrixBase::getIndexing() const {
        return indexing_;
    }

    bool BlockMatrixBase::isSymmetric() const {
        return symmetric_;
    }
} // finalicp