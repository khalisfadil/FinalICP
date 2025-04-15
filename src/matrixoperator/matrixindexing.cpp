#include <matrixoperator/matrixindexing.hpp>

#include <stdexcept>
#include <iostream>

namespace finalicp {
    
    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------

    BlockDimIndexing::BlockDimIndexing() {}

    BlockDimIndexing::BlockDimIndexing(const std::vector<unsigned int>& blkSizes)
        : blkSizes_(blkSizes) {

        // Check input has entries
        if (blkSizes_.empty()) {
            throw std::invalid_argument("[BlockDimIndexing] Tried to initialize a block matrix with no size.");
        }

        // Initialize scalar size and cumulative entries
        scalarDim_ = 0;
        cumBlkSizes_.resize(blkSizes_.size());
        unsigned int i = 0;
        for (const auto& blockSize : blkSizes_) {
            // Check that each input has a valid size
            if (blockSize == 0) {
                throw std::invalid_argument("[BlockDimIndexing] Tried to initialize a block row size of 0.");
            }
            // Add up cumulative sizes
            cumBlkSizes_[i] = scalarDim_;
            scalarDim_ += blockSize;
            i++;
        }
    }

    // -----------------------------------------------------------------------------
    // getBlockSizes
    // -----------------------------------------------------------------------------

    const std::vector<unsigned int>& BlockDimIndexing::blkSizes() const {
        return blkSizes_;
    }

    // -----------------------------------------------------------------------------
    // numEntries
    // -----------------------------------------------------------------------------

    unsigned int BlockDimIndexing::numEntries() const {
        return blkSizes_.size();
    }

    // -----------------------------------------------------------------------------
    // blkSizeAt
    // -----------------------------------------------------------------------------

    unsigned int BlockDimIndexing::blkSizeAt(unsigned int index) const {
        return blkSizes_.at(index);
    }

    // -----------------------------------------------------------------------------
    // cumSumAt
    // -----------------------------------------------------------------------------

    unsigned int BlockDimIndexing::cumSumAt(unsigned int index) const {
        return cumBlkSizes_.at(index);
    }

    // -----------------------------------------------------------------------------
    // scalarSize
    // -----------------------------------------------------------------------------

    unsigned int BlockDimIndexing::scalarSize() const {
        return scalarDim_;
    }

    // -----------------------------------------------------------------------------
    // BlockMatrixIndexing
    // -----------------------------------------------------------------------------

    BlockMatrixIndexing::BlockMatrixIndexing()
        : blkSizeSymmetric_(false) {
    }

    // -----------------------------------------------------------------------------
    // BlockMatrixIndexing
    // -----------------------------------------------------------------------------

    BlockMatrixIndexing::BlockMatrixIndexing(const std::vector<unsigned int>& blkSizes)
        : blkRowIndexing_(blkSizes), blkSizeSymmetric_(true) {
    }

    // -----------------------------------------------------------------------------
    // BlockMatrixIndexing
    // -----------------------------------------------------------------------------

    BlockMatrixIndexing::BlockMatrixIndexing(const std::vector<unsigned int>& blkRowSizes,
                        const std::vector<unsigned int>& blkColSizes)
        : blkRowIndexing_(blkRowSizes), blkColIndexing_(blkColSizes), blkSizeSymmetric_(false) {
    }

    // -----------------------------------------------------------------------------
    // rowIndexing
    // -----------------------------------------------------------------------------

    const BlockDimIndexing& BlockMatrixIndexing::rowIndexing() const {
        return blkRowIndexing_;
    }

    // -----------------------------------------------------------------------------
    // colIndexing
    // -----------------------------------------------------------------------------

    const BlockDimIndexing& BlockMatrixIndexing::colIndexing() const {
        if (!blkSizeSymmetric_) {
            return blkColIndexing_;
        } else {
            return blkRowIndexing_;
        }
    }

} // finalicp