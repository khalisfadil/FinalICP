#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace finalicp{

    //Manages 1D indexing for block dimensions (rows or columns). 
    class BlockDimIndexing {
        public:
            //Default constructor. Initializes an empty indexing structure.
            BlockDimIndexing();

            //Constructor to initialize block sizes and compute cumulative sizes and total size.
            BlockDimIndexing(const std::vector<unsigned int>& blkSqSizes);

            //Returns the vector of block sizes.
            const std::vector<unsigned int>& blkSizes() const;

            //Returns the total number of blocks entries.
            unsigned int numEntries() const;

            //Returns the size of a block at the given index.
            unsigned int blkSizeAt(unsigned int index) const;

            //Returns the cumulative offset of a block at the given index.
            unsigned int cumSumAt(unsigned int index) const;

            //Get scalar size
            unsigned int scalarSize() const;

        private:
            //Stores the sizes of each block (e.g., row or column sizes).
            std::vector<unsigned int> blkSizes_;

            //Stores the cumulative block sizes (offsets) for each block.
            std::vector<unsigned int> cumBlkSizes_;

            //Stores the total scalar size (sum of all block sizes).
            unsigned int scalarDim_;

    };
    
    //Manages block indexing for rows and columns in a block matrix.
    class BlockMatrixIndexing {

        public:
            //Default constructor. Initializes an empty block matrix indexing structure.
            BlockMatrixIndexing();

            //Constructor for a **symmetric block matrix**.
            BlockMatrixIndexing(const std::vector<unsigned int>& blkSizes);

            //Rectangular matrix constructor
            BlockMatrixIndexing(const std::vector<unsigned int>& blkRowSizes,
                      const std::vector<unsigned int>& blkColSizes);

            //Returns the row indexing information.
            const BlockDimIndexing& rowIndexing() const;
            
            //Returns the column indexing information.
            const BlockDimIndexing& colIndexing() const;

        private:
            //Block row indexing
            BlockDimIndexing blkRowIndexing_;

            //Block column indexing
            BlockDimIndexing blkColIndexing_;

            //Whether matrix is block-size-symmetric
            bool blkSizeSymmetric_;
    };
}