#include <matrixoperator/matrixsparse.hpp>

#include <stdexcept>
#include <iostream>

namespace finalicp{

    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------
    
    BlockSparseMatrix::BlockSparseMatrix() : BlockMatrixBase() {} 

    BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blkRowSizes,
                                        const std::vector<unsigned int>& blkColSizes)
        : BlockMatrixBase(blkRowSizes, blkColSizes) {
        // Setup data structures
        cols_.clear();
        cols_.resize(getIndexing().colIndexing().numEntries());
    }

    BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blkSizes, bool symmetric)
        : BlockMatrixBase(blkSizes, symmetric) {
        // Setup data structures
        cols_.clear();
        cols_.resize(getIndexing().colIndexing().numEntries());
    }

    // -----------------------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------------------

    void BlockSparseMatrix::clear() {
        for (unsigned int c = 0; c < this->getIndexing().colIndexing().numEntries(); c++) {
            cols_[c].rows.clear();
        }
    }


    void BlockSparseMatrix::zero() {
        for (unsigned int c = 0; c < this->getIndexing().colIndexing().numEntries(); c++) {
            for(std::map<unsigned int, BlockRowEntry>::iterator it = cols_[c].rows.begin();
                it != cols_[c].rows.end(); ++it) {
                it->second.data.setZero();
            }
        }
    }

    void BlockSparseMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
        // Get references to indexing objects
        const BlockDimIndexing& blkRowIndexing = this->getIndexing().rowIndexing();
        const BlockDimIndexing& blkColIndexing = this->getIndexing().colIndexing();

        // Check if indexing is valid
        if (r >= blkRowIndexing.numEntries() || c >= blkColIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::add] Invalid row or column index for block structure.");
        }

        // Early symmetric check to avoid unnecessary work
        if (this->isSymmetric() && r > c) {
            std::cout << "[BlockSparseMatrix::add] Ignored add operation to lower half of upper-symmetric block-sparse matrix." << std::endl;
            return;
        }

        // Validate matrix dimensions
        if (m.rows() != static_cast<int>(blkRowIndexing.blkSizeAt(r)) ||
            m.cols() != static_cast<int>(blkColIndexing.blkSizeAt(c))) {
            throw std::invalid_argument("[BlockSparseMatrix::add] Matrix dimensions do not match block structure at row " +
                                        std::to_string(r) + ", col " + std::to_string(c));
        }

        // Direct insertion or update using try_emplace for efficiency
        auto& rowMap = cols_[c].rows;
        auto [it, inserted] = rowMap.try_emplace(r, BlockRowEntry());
        if (inserted) {
            it->second.data = m; // New entry
        } else {
            it->second.data += m; // Add to existing entry
        }
    }

    BlockSparseMatrix::BlockRowEntry& BlockSparseMatrix::rowEntryAt(unsigned int r, unsigned int c, bool allowInsert) {
        // Cache indexing objects
        const BlockDimIndexing& rowIndexing = this->getIndexing().rowIndexing();
        const BlockDimIndexing& colIndexing = this->getIndexing().colIndexing();

        // Validate indices
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::rowEntryAt] Invalid row or column index for block structure.");
        }

        // Check symmetric matrix constraints
        if (this->isSymmetric() && r > c) {
            std::cout << "[BlockSparseMatrix::rowEntryAt] Cannot access lower half of upper-symmetric block-sparse matrix." << std::endl;
            throw std::invalid_argument("Invalid access to lower-triangular block in symmetric matrix.");
        }

        // Access column
        BlockSparseColumn& colRef = cols_[c];

        // Try to find or insert row entry
        auto& rowMap = colRef.rows;
        if (allowInsert) {
            auto [it, inserted] = rowMap.try_emplace(r, BlockRowEntry());
            if (inserted) {
                it->second.data = Eigen::MatrixXd::Zero(rowIndexing.blkSizeAt(r), colIndexing.blkSizeAt(c));
            }
            return it->second;
        } else {
            auto it = rowMap.find(r);
            if (it == rowMap.end()) {
                throw std::invalid_argument("[BlockSparseMatrix::rowEntryAt] Requested block at (" + std::to_string(r) + ", " + std::to_string(c) + ") does not exist.");
            }
            return it->second;
        }
    }

    Eigen::MatrixXd& BlockSparseMatrix::at(unsigned int r, unsigned int c) {
        // Cache indexing objects
        const BlockDimIndexing& rowIndexing = this->getIndexing().rowIndexing();
        const BlockDimIndexing& colIndexing = this->getIndexing().colIndexing();

        // Validate indices
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::at] Invalid row or column index for block structure.");
        }

        // Check symmetric matrix constraints
        if (this->isSymmetric() && r > c) {
            throw std::invalid_argument("[BlockSparseMatrix::at] Invalid access to lower-triangular block in symmetric matrix at (" +
                                        std::to_string(r) + ", " + std::to_string(c) + ").");
        }

        // Access column and find row entry
        auto& rowMap = cols_[c].rows;
        auto it = rowMap.find(r);
        if (it == rowMap.end()) {
            throw std::invalid_argument("[BlockSparseMatrix::at] Block at (" + std::to_string(r) + ", " + std::to_string(c) + ") does not exist.");
        }

        return it->second.data;
    }

    Eigen::MatrixXd BlockSparseMatrix::copyAt(unsigned int r, unsigned int c) const {
        // Cache indexing objects
        const BlockDimIndexing& rowIndexing = this->getIndexing().rowIndexing();
        const BlockDimIndexing& colIndexing = this->getIndexing().colIndexing();

        // Validate indices
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::copyAt] Invalid row or column index for block structure.");
        }

        // Determine which block to access (symmetric case uses (c, r) for lower triangle)
        unsigned int colIdx = this->isSymmetric() && r > c ? r : c;
        unsigned int rowIdx = this->isSymmetric() && r > c ? c : r;

        // Find row entry
        auto& rowMap = cols_[colIdx].rows;
        auto it = rowMap.find(rowIdx);
        if (it == rowMap.end()) {
            return Eigen::MatrixXd::Zero(rowIndexing.blkSizeAt(r), colIndexing.blkSizeAt(c));
        }

        // Return data (transpose for symmetric lower-triangular access)
        return (this->isSymmetric() && r > c) ? it->second.data.transpose() : it->second.data;
    }

    Eigen::SparseMatrix<double> BlockSparseMatrix::toEigen(bool getSubBlockSparsity) const {
        // Cache indexing objects
        const BlockDimIndexing& rowIndexing = this->getIndexing().rowIndexing();
        const BlockDimIndexing& colIndexing = this->getIndexing().colIndexing();

        // Initialize sparse matrix
        Eigen::SparseMatrix<double> mat(rowIndexing.scalarSize(), colIndexing.scalarSize());
        mat.reserve(this->getNnzPerCol());

        // Use triplet list for efficient insertion
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(getNnzPerCol().sum()); // Approximate capacity

        // Iterate over block-sparse columns
        for (unsigned int c = 0; c < colIndexing.numEntries(); ++c) {
            unsigned int colBlkSize = colIndexing.blkSizeAt(c);
            unsigned int colCumSum = colIndexing.cumSumAt(c);

            // Iterate over non-zero rows in the column
            for (const auto& [r, entry] : cols_[c].rows) {
                unsigned int rowBlkSize = rowIndexing.blkSizeAt(r);
                unsigned int rowCumSum = rowIndexing.cumSumAt(r);

                // Skip lower-triangular blocks for symmetric matrices
                if (this->isSymmetric() && r > c) {
                    continue;
                }

                // Iterate over block elements in column-major order
                for (unsigned int j = 0; j < colBlkSize; ++j) {
                    for (unsigned int i = 0; i < rowBlkSize; ++i) {
                        double v_ij = entry.data(i, j);
                        if (!getSubBlockSparsity || std::abs(v_ij) > 1e-15) {
                            unsigned int rowIdx = rowCumSum + i;
                            unsigned int colIdx = colCumSum + j;
                            triplets.emplace_back(rowIdx, colIdx, v_ij);

                            // Add mirrored entry for symmetric matrices
                            if (this->isSymmetric() && r != c) {
                                triplets.emplace_back(colCumSum + j, rowCumSum + i, v_ij);
                            }
                        }
                    }
                }
            }
        }

        // Build sparse matrix from triplets
        mat.setFromTriplets(triplets.begin(), triplets.end());
        mat.makeCompressed();

        return mat;
    }

    Eigen::VectorXi BlockSparseMatrix::getNnzPerCol() const {
        // Cache column indexing
        const BlockDimIndexing& colIndexing = this->getIndexing().colIndexing();

        // Initialize result vector
        Eigen::VectorXi result = Eigen::VectorXi::Zero(colIndexing.scalarSize());

        // Iterate over block columns
        for (unsigned int c = 0; c < colIndexing.numEntries(); ++c) {
            unsigned int colBlkSize = colIndexing.blkSizeAt(c);
            unsigned int colCumSum = colIndexing.cumSumAt(c);

            // Count non-zero scalar rows in this block column
            unsigned int nnz = 0;
            for (const auto& [r, entry] : cols_[c].rows) {
                nnz += entry.data.rows();
            }

            // Assign nnz to all scalar columns in this block
            result.segment(colCumSum, colBlkSize).setConstant(nnz);
        }

        return result;
    }

} // namespace finalicp