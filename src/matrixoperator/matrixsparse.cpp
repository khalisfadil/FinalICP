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
        cols_.resize(getIndexing().colIndexing().numEntries());
    }

    BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blkSizes, bool symmetric)
        : BlockMatrixBase(blkSizes, symmetric) {
        // Setup data structures
        cols_.resize(getIndexing().colIndexing().numEntries());
    }

    // -----------------------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------------------

    void BlockSparseMatrix::clear() {
        tbb::parallel_for(
            tbb::blocked_range<unsigned int>(0, getIndexing().colIndexing().numEntries()),
            [&](const tbb::blocked_range<unsigned int>& range) {
                for (unsigned int c = range.begin(); c != range.end(); ++c) {
                    cols_[c].rows.clear();
                }
            });
    }

    void BlockSparseMatrix::zero() {
        tbb::parallel_for(
            tbb::blocked_range<unsigned int>(0, getIndexing().colIndexing().numEntries()),
            [&](const tbb::blocked_range<unsigned int>& range) {
                for (unsigned int c = range.begin(); c != range.end(); ++c) {
                    for (auto it = cols_[c].rows.begin(); it != cols_[c].rows.end(); ++it) {
                        it->second.data.setZero();
                    }
                }
            });
    }

    void BlockSparseMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
        // Cache indexing object
        const BlockMatrixIndexing& indexing = getIndexing();
        const BlockDimIndexing& rowIndexing = indexing.rowIndexing();
        const BlockDimIndexing& colIndexing = indexing.colIndexing();

        // Validate indices
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::add] Block indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of bounds");
        }

        // Enforce upper-triangular for symmetric matrices
        if (isSymmetric() && r > c) {
            return; // Silently ignore for efficiency
        }

        // Validate block dimensions
        if (m.rows() != static_cast<int>(rowIndexing.blkSizeAt(r)) ||
            m.cols() != static_cast<int>(colIndexing.blkSizeAt(c))) {
            throw std::invalid_argument("[BlockSparseMatrix::add] Block at (" + std::to_string(r) + ", " + std::to_string(c) +
                                    ") has incorrect size: expected (" +
                                    std::to_string(rowIndexing.blkSizeAt(r)) + ", " +
                                    std::to_string(colIndexing.blkSizeAt(c)) + "), got (" +
                                    std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ")");
        }

        // Thread-safe insert or update
        tbb::concurrent_hash_map<unsigned int, BlockRowEntry>::accessor acc;
        if (cols_[c].rows.insert(acc, r)) {
            acc->second.data = m; // New entry
        } else {
            acc->second.data += m; // Accumulate
        }
    }

    BlockSparseMatrix::BlockRowEntry& BlockSparseMatrix::rowEntryAt(unsigned int r, unsigned int c, bool allowInsert) {
        // Cache indexing object
        const BlockMatrixIndexing& indexing = getIndexing();
        const BlockDimIndexing& rowIndexing = indexing.rowIndexing();
        const BlockDimIndexing& colIndexing = indexing.colIndexing();

        // Validate indices
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::rowEntryAt] Block indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of bounds");
        }

        // Enforce upper-triangular for symmetric matrices
        if (isSymmetric() && r > c) {
            throw std::invalid_argument("[BlockSparseMatrix::rowEntryAt] Cannot access lower triangle of symmetric matrix at (" +
                                    std::to_string(r) + ", " + std::to_string(c) + ")");
        }

        // Thread-safe access or insert
        tbb::concurrent_hash_map<unsigned int, BlockRowEntry>::accessor acc;
        if (cols_[c].rows.find(acc, r)) {
            return acc->second; // Existing entry
        }
        if (!allowInsert) {
            throw std::invalid_argument("[BlockSparseMatrix::rowEntryAt] Block at (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") does not exist");
        }

        // Insert new zero-initialized entry
        cols_[c].rows.insert(acc, r);
        acc->second.data = Eigen::MatrixXd::Zero(rowIndexing.blkSizeAt(r), colIndexing.blkSizeAt(c));
        return acc->second;
    }

    Eigen::MatrixXd& BlockSparseMatrix::at(unsigned int r, unsigned int c) {
        // Cache indexing object
        const BlockMatrixIndexing& indexing = getIndexing();
        const BlockDimIndexing& rowIndexing = indexing.rowIndexing();
        const BlockDimIndexing& colIndexing = indexing.colIndexing();

        // Validate indices
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::at] Block indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of bounds");
        }

        // Enforce upper-triangular for symmetric matrices
        if (isSymmetric() && r > c) {
            throw std::invalid_argument("[BlockSparseMatrix::at] Cannot access lower triangle of symmetric matrix at (" +
                                    std::to_string(r) + ", " + std::to_string(c) + ")");
        }

        // Thread-safe access
        tbb::concurrent_hash_map<unsigned int, BlockRowEntry>::accessor acc;
        if (!cols_[c].rows.find(acc, r)) {
            throw std::invalid_argument("[BlockSparseMatrix::at] Block at (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") does not exist");
        }

        return acc->second.data;
    }

    Eigen::MatrixXd BlockSparseMatrix::copyAt(unsigned int r, unsigned int c) const {
        // Cache indexing object
        const BlockMatrixIndexing& indexing = getIndexing();
        const BlockDimIndexing& rowIndexing = indexing.rowIndexing();
        const BlockDimIndexing& colIndexing = indexing.colIndexing();

        // Validate indices
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument("[BlockSparseMatrix::copyAt] Block indices (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") out of bounds");
        }

        // Prepare zero matrix for non-existing entries
        Eigen::MatrixXd zeroMatrix = Eigen::MatrixXd::Zero(rowIndexing.blkSizeAt(r),
                                                        colIndexing.blkSizeAt(c));

        // Handle symmetric lower-triangular access
        if (isSymmetric() && r > c) {
            tbb::concurrent_hash_map<unsigned int, BlockRowEntry>::const_accessor acc;
            if (cols_[r].rows.find(acc, c)) {
                return acc->second.data.transpose();
            }
            return zeroMatrix;
        }

        // Handle non-symmetric or upper-triangular access
        tbb::concurrent_hash_map<unsigned int, BlockRowEntry>::const_accessor acc;
        if (cols_[c].rows.find(acc, r)) {
            return acc->second.data;
        }
        return zeroMatrix;
    }

    Eigen::VectorXi BlockSparseMatrix::getNnzPerCol() const {
        // Cache indexing object
        const BlockMatrixIndexing& indexing = getIndexing();
        const BlockDimIndexing& colIndexing = indexing.colIndexing();

        // Allocate result vector
        Eigen::VectorXi result(colIndexing.scalarSize());

        // Parallel computation of non-zero counts
        tbb::parallel_for(
            tbb::blocked_range<unsigned int>(0, colIndexing.numEntries()),
            [&](const tbb::blocked_range<unsigned int>& range) {
                for (unsigned int c = range.begin(); c != range.end(); ++c) {
                    unsigned int nnz = 0;
                    for (auto it = cols_[c].rows.begin(); it != cols_[c].rows.end(); ++it) {
                        nnz += it->second.data.rows();
                    }
                    result.segment(colIndexing.cumSumAt(c), colIndexing.blkSizeAt(c)).setConstant(nnz);
                }
            });

        return result;
    }

    Eigen::SparseMatrix<double> BlockSparseMatrix::toEigen(bool getSubBlockSparsity) const {
        // Cache indexing objects
        const BlockMatrixIndexing& indexing = getIndexing();
        const BlockDimIndexing& rowIndexing = indexing.rowIndexing();
        const BlockDimIndexing& colIndexing = indexing.colIndexing();

        // Allocate sparse matrix
        Eigen::SparseMatrix<double> mat(rowIndexing.scalarSize(), colIndexing.scalarSize());
        mat.reserve(getNnzPerCol());

        // Collect triplets in parallel
        tbb::concurrent_vector<std::tuple<unsigned int, unsigned int, double>> triplets;
        tbb::parallel_for(
            tbb::blocked_range<unsigned int>(0, colIndexing.numEntries()),
            [&](const tbb::blocked_range<unsigned int>& range) {
                for (unsigned int c = range.begin(); c != range.end(); ++c) {
                    unsigned int colBlkSize = colIndexing.blkSizeAt(c);
                    unsigned int colCumSum = colIndexing.cumSumAt(c);
                    for (auto it = cols_[c].rows.begin(); it != cols_[c].rows.end(); ++it) {
                        unsigned int r = it->first;
                        unsigned int rowBlkSize = rowIndexing.blkSizeAt(r);
                        unsigned int rowCumSum = rowIndexing.cumSumAt(r);

                        unsigned int colIdx = colCumSum;
                        for (unsigned int j = 0; j < colBlkSize; ++j, ++colIdx) {
                            unsigned int rowIdx = rowCumSum;
                            for (unsigned int i = 0; i < rowBlkSize; ++i, ++rowIdx) {
                                double v_ij = it->second.data(i, j);
                                if (std::fabs(v_ij) > 1e-10 || !getSubBlockSparsity) {
                                    triplets.emplace_back(rowIdx, colIdx, v_ij);
                                    // Add lower-triangular entry for symmetric matrices
                                    if (isSymmetric() && r != c) {
                                        triplets.emplace_back(colIdx, rowIdx, v_ij);
                                    }
                                }
                            }
                        }
                    }
                }
            });

        // Insert triplets sequentially
        for (const auto& [row, col, val] : triplets) {
            mat.insert(row, col) = val;
        }

        // Compress into compact format
        mat.makeCompressed();

        return mat;
    }

} // namespace finalicp