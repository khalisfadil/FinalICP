#include <matrixoperator/matrix.hpp>

#include <stdexcept>
#include <iostream>

namespace finalicp {
    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------

    BlockMatrix::BlockMatrix() : BlockMatrixBase() {}

    BlockMatrix::BlockMatrix(const std::vector<unsigned int>& blkRowSizes,
                            const std::vector<unsigned int>& blkColSizes)
        : BlockMatrixBase(blkRowSizes, blkColSizes),
        data_(getIndexing().rowIndexing().numEntries(),
                std::vector<Eigen::MatrixXd>(getIndexing().colIndexing().numEntries())) {
        zero();
    }

    BlockMatrix::BlockMatrix(const std::vector<unsigned int>& blkSizes, bool symmetric)
        : BlockMatrixBase(blkSizes, symmetric),
        data_(getIndexing().rowIndexing().numEntries(),
                std::vector<Eigen::MatrixXd>(getIndexing().colIndexing().numEntries())) {
        zero();
    }

    // -----------------------------------------------------------------------------
    // Application
    // -----------------------------------------------------------------------------

    void BlockMatrix::zero() {
        const auto& rowIndexing = getIndexing().rowIndexing();
        const auto& colIndexing = getIndexing().colIndexing();
        
        for (unsigned int r = 0; r < data_.size(); ++r) {
            auto rowSize = rowIndexing.blkSizeAt(r);
            for (unsigned int c = 0; c < data_[r].size(); ++c) {
                data_[r][c].resize(rowSize, colIndexing.blkSizeAt(c));
                data_[r][c].setZero();
            }
        }
    }

    void BlockMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
        const auto& rowIndexing = getIndexing().rowIndexing();
        const auto& colIndexing = getIndexing().colIndexing();

        // Check index validity
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument(
                "[BlockMatrix::add] Invalid row or column index: (" + std::to_string(r) + ", " +
                std::to_string(c) + ") exceeds (" + std::to_string(rowIndexing.numEntries()) + ", " +
                std::to_string(colIndexing.numEntries()) + ").");
        }

        // Check symmetry for upper-triangular access
        if (isSymmetric() && r > c) {
            std::cerr << "[BlockMatrix::add] Ignoring add to lower triangle of symmetric matrix at (" 
                    << r << ", " << c << ")." << std::endl;
            return;
        }

        // Check matrix dimensions
        auto rowSize = rowIndexing.blkSizeAt(r);
        auto colSize = colIndexing.blkSizeAt(c);
        if (m.rows() != static_cast<int>(rowSize) || m.cols() != static_cast<int>(colSize)) {
            throw std::invalid_argument(
                "[BlockMatrix] Matrix size (" + std::to_string(m.rows()) + "x" +
                std::to_string(m.cols()) + ") at (" + std::to_string(r) + ", " + std::to_string(c) +
                ") does not match block size (" + std::to_string(rowSize) + "x" +
                std::to_string(colSize) + ").");
        }

        // Add matrix
        data_[r][c] += m;
    }

    Eigen::MatrixXd& BlockMatrix::at(unsigned int r, unsigned int c) {
        const auto& rowIndexing = getIndexing().rowIndexing();
        const auto& colIndexing = getIndexing().colIndexing();

        // Check index validity
        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument(
                "[BlockMatrix::at(] Invalid index: (" + std::to_string(r) + ", " +
                std::to_string(c) + ") exceeds (" + std::to_string(rowIndexing.numEntries()) +
                ", " + std::to_string(colIndexing.numEntries()) + ").");
        }

        // Check symmetry for upper-triangular access
        if (isSymmetric() && r > c) {
            throw std::runtime_error(
                "[BlockMatrix::at(] Cannot return reference to lower triangle at (" +
                std::to_string(r) + ", " + std::to_string(c) + ") in symmetric matrix.");
        }

        return data_[r][c];
    }

    Eigen::MatrixXd BlockMatrix::copyAt(unsigned int r, unsigned int c) const {
        const auto& rowIndexing = getIndexing().rowIndexing();
        const auto& colIndexing = getIndexing().colIndexing();

        if (r >= rowIndexing.numEntries() || c >= colIndexing.numEntries()) {
            throw std::invalid_argument(
                "[BlockMatrix::copyAt] Invalid index: (" + std::to_string(r) + ", " +
                std::to_string(c) + ") exceeds (" + std::to_string(rowIndexing.numEntries()) +
                ", " + std::to_string(colIndexing.numEntries()) + ").");
        }

        return isSymmetric() && r > c ? data_[c][r].transpose() : data_[r][c];
    }

} // finalicp