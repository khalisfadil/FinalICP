#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <matrixoperator/matrixsparse.hpp>
#include <matrixoperator/vector.hpp>
#include <matrixoperator/matrix.hpp>

namespace finalicp {

class BlockSparseMatrixVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator for stress tests
        rng_.seed(42);
        dist_ = std::uniform_real_distribution<double>(-1.0, 1.0);

        // Common block sizes for tests
        block_sizes_ = {2, 3, 1}; // 2x2, 3x3, 1x1 blocks
        scalar_size_ = 2 + 3 + 1; // Total scalar size = 6

        // Sample matrices for testing
        m_diag_2x2_ << 2, 1, 1, 2;
        m_offdiag_2x3_ << 0, 0, 0.1, 0, 0, 0;
        m_diag_3x3_ << 3, 1, 0, 1, 3, 1, 0, 1, 3;
        m_ones_2x2_ = Eigen::MatrixXd::Ones(2, 2);
        m_ones_3x3_ = Eigen::MatrixXd::Ones(3, 3);
        m_zero_2x3_ = Eigen::MatrixXd::Zero(2, 3);
        m_zero_3x1_ = Eigen::MatrixXd::Zero(3, 1);

        // Sample vector for testing
        b_ << 1, 2, 3, 4, 5, 6;
    }

    // Helper to print matrix pattern
    void PrintMatrixPattern(const Eigen::SparseMatrix<double>& mat, const std::string& name) {
        std::cout << "\n=== " << name << " ===\n";
        std::cout << mat << "\n";
        std::cout << "Non-zeros: " << mat.nonZeros() << "\n";
    }

    // Helper to print vector
    void PrintVector(const BlockVector& vec, const std::string& name) {
        std::cout << "\n=== " << name << " ===\n";
        std::cout << vec.toEigen().transpose() << "\n";
    }

    // Helper to generate random matrix
    Eigen::MatrixXd RandomMatrix(int rows, int cols) {
        Eigen::MatrixXd m(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m(i, j) = dist_(rng_);
            }
        }
        return m;
    }

    // Helper to generate symmetric positive definite matrix
    Eigen::MatrixXd RandomSPDMatrix(int size) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(size, size);
        Eigen::MatrixXd spd = A * A.transpose() + 10.0 * Eigen::MatrixXd::Identity(size, size);
        return spd;
    }

    // Helper to convert BlockMatrix to Eigen sparse matrix
    Eigen::SparseMatrix<double> toEigen(const BlockMatrix& mat) const {
        const auto& rowIndexing = mat.getIndexing().rowIndexing();
        const auto& colIndexing = mat.getIndexing().colIndexing();
        Eigen::SparseMatrix<double> eig_mat(rowIndexing.scalarSize(), colIndexing.scalarSize());
        std::vector<Eigen::Triplet<double>> triplets;

        for (unsigned int r = 0; r < rowIndexing.numEntries(); ++r) {
            unsigned int rowBlkSize = rowIndexing.blkSizeAt(r);
            unsigned int rowCumSum = rowIndexing.cumSumAt(r);
            for (unsigned int c = 0; c < colIndexing.numEntries(); ++c) {
                unsigned int colBlkSize = colIndexing.blkSizeAt(c);
                unsigned int colCumSum = colIndexing.cumSumAt(c);
                const Eigen::MatrixXd& block = mat.copyAt(r, c);
                for (unsigned int i = 0; i < rowBlkSize; ++i) {
                    for (unsigned int j = 0; j < colBlkSize; ++j) {
                        double value = block(i, j);
                        if (std::abs(value) > 1e-15) {
                            triplets.emplace_back(rowCumSum + i, colCumSum + j, value);
                        }
                    }
                }
            }
        }

        eig_mat.setFromTriplets(triplets.begin(), triplets.end());
        eig_mat.makeCompressed();
        return eig_mat;
    }

    // Random number generator
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;

    // Common test data
    std::vector<unsigned int> block_sizes_;
    unsigned int scalar_size_;
    Eigen::MatrixXd m_diag_2x2_{2, 2};
    Eigen::MatrixXd m_offdiag_2x3_{2, 3};
    Eigen::MatrixXd m_diag_3x3_{3, 3};
    Eigen::MatrixXd m_ones_2x2_{2, 2};
    Eigen::MatrixXd m_ones_3x3_{3, 3};
    Eigen::MatrixXd m_zero_2x3_{2, 3};
    Eigen::MatrixXd m_zero_3x1_{3, 1};
    Eigen::VectorXd b_{6};
};

// Test BlockMatrix basic functionality
TEST_F(BlockSparseMatrixVectorTest, BlockMatrixOperations) {
    std::cout << "\n=== Testing BlockMatrix Operations ===\n";

    // Test non-symmetric matrix
    {
        BlockMatrix mat(block_sizes_, block_sizes_); // Non-symmetric
        mat.add(0, 0, m_diag_2x2_);
        mat.add(0, 1, m_offdiag_2x3_);
        mat.add(1, 1, m_diag_3x3_);
        PrintMatrixPattern(toEigen(mat), "Non-symmetric: After adding blocks");

        // Test at
        EXPECT_TRUE(mat.at(0, 0).isApprox(m_diag_2x2_));
        EXPECT_TRUE(mat.at(0, 1).isApprox(m_offdiag_2x3_));
        EXPECT_TRUE(mat.at(1, 1).isApprox(m_diag_3x3_));
        EXPECT_TRUE(mat.at(2, 2).isApprox(Eigen::MatrixXd::Zero(1, 1))); // Unset block

        // Test copyAt
        EXPECT_TRUE(mat.copyAt(0, 0).isApprox(m_diag_2x2_));
        EXPECT_TRUE(mat.copyAt(0, 1).isApprox(m_offdiag_2x3_));
        EXPECT_TRUE(mat.copyAt(1, 0).isApprox(Eigen::MatrixXd::Zero(3, 2))); // Unset block

        // Test zero
        mat.zero();
        PrintMatrixPattern(toEigen(mat), "Non-symmetric: After zero");
        EXPECT_TRUE(mat.at(0, 0).isApprox(Eigen::MatrixXd::Zero(2, 2)));
        EXPECT_TRUE(mat.at(0, 1).isApprox(Eigen::MatrixXd::Zero(2, 3)));
    }

    // Test symmetric matrix
    {
        BlockMatrix mat(block_sizes_, true); // Symmetric
        mat.add(0, 0, m_diag_2x2_);
        mat.add(0, 1, m_offdiag_2x3_);
        mat.add(1, 1, m_diag_3x3_);
        PrintMatrixPattern(toEigen(mat), "Symmetric: After adding blocks");

        // Test at (upper triangle only)
        EXPECT_TRUE(mat.at(0, 0).isApprox(m_diag_2x2_));
        EXPECT_TRUE(mat.at(0, 1).isApprox(m_offdiag_2x3_));
        EXPECT_TRUE(mat.at(1, 1).isApprox(m_diag_3x3_));
        EXPECT_THROW(mat.at(1, 0), std::runtime_error); // Lower triangle

        // Test copyAt (mirrors lower triangle)
        EXPECT_TRUE(mat.copyAt(0, 0).isApprox(m_diag_2x2_));
        EXPECT_TRUE(mat.copyAt(0, 1).isApprox(m_offdiag_2x3_));
        EXPECT_TRUE(mat.copyAt(1, 0).isApprox(m_offdiag_2x3_.transpose()));
        EXPECT_TRUE(mat.copyAt(2, 2).isApprox(Eigen::MatrixXd::Zero(1, 1)));

        // Test zero
        mat.zero();
        PrintMatrixPattern(toEigen(mat), "Symmetric: After zero");
        EXPECT_TRUE(mat.at(0, 0).isApprox(Eigen::MatrixXd::Zero(2, 2)));
    }

    // Test invalid inputs
    {
        BlockMatrix mat(block_sizes_, true);
        EXPECT_THROW(mat.add(3, 0, m_diag_2x2_), std::invalid_argument); // Invalid index
        EXPECT_THROW(mat.at(0, 3), std::invalid_argument); // Invalid index
        EXPECT_THROW(mat.copyAt(3, 0), std::invalid_argument); // Invalid index
        Eigen::MatrixXd wrong_size(2, 3);
        EXPECT_THROW(mat.add(0, 0, wrong_size), std::invalid_argument); // Wrong size
        EXPECT_NO_THROW(mat.add(1, 0, m_zero_2x3_)); // Ignored (symmetric lower triangle)
    }
}

// Test BlockSparseMatrix basic functionality
TEST_F(BlockSparseMatrixVectorTest, BasicMatrixOperations) {
    std::cout << "\n=== Testing Basic Matrix Operations ===\n";
    BlockSparseMatrix mat(block_sizes_, true); // Symmetric matrix

    // Test add
    mat.add(0, 0, m_diag_2x2_);
    mat.add(0, 1, m_offdiag_2x3_);
    mat.add(1, 1, m_diag_3x3_);
    PrintMatrixPattern(mat.toEigen(true), "After adding blocks");

    // Test at
    EXPECT_TRUE(mat.at(0, 0).isApprox(m_diag_2x2_));
    EXPECT_TRUE(mat.at(0, 1).isApprox(m_offdiag_2x3_));
    EXPECT_TRUE(mat.at(1, 1).isApprox(m_diag_3x3_));

    // Test copyAt
    EXPECT_TRUE(mat.copyAt(0, 0).isApprox(m_diag_2x2_));
    EXPECT_TRUE(mat.copyAt(1, 0).isApprox(m_offdiag_2x3_.transpose())); // Symmetric case
    EXPECT_TRUE(mat.copyAt(2, 2).isApprox(Eigen::MatrixXd::Zero(1, 1))); // Non-existent block

    // Test rowEntryAt
    auto& entry = mat.rowEntryAt(0, 0, false);
    EXPECT_TRUE(entry.data.isApprox(m_diag_2x2_));

    // Test clear
    mat.clear();
    PrintMatrixPattern(mat.toEigen(true), "After clear");
    EXPECT_EQ(mat.toEigen(true).nonZeros(), 0);

    // Test zero
    mat.add(0, 0, m_diag_2x2_);
    mat.zero();
    PrintMatrixPattern(mat.toEigen(true), "After zero");
    EXPECT_TRUE(mat.at(0, 0).isApprox(Eigen::MatrixXd::Zero(2, 2)));
}

// Test BlockVector basic functionality
TEST_F(BlockSparseMatrixVectorTest, BasicVectorOperations) {
    std::cout << "\n=== Testing Basic Vector Operations ===\n";
    BlockVector vec(block_sizes_);

    // Test add
    Eigen::VectorXd v2(2), v3(3), v1(1);
    v2 << 1, 2;
    v3 << 3, 4, 5;
    v1 << 6;
    vec.add(0, v2);
    vec.add(1, v3);
    vec.add(2, v1);
    PrintVector(vec, "After adding blocks");

    // Test at
    EXPECT_TRUE(vec.at(0).isApprox(v2));
    EXPECT_TRUE(vec.at(1).isApprox(v3));
    EXPECT_TRUE(vec.at(2).isApprox(v1));

    // Test mapAt (read-only)
    {
        const BlockVector& const_vec = vec;
        Eigen::Map<const Eigen::VectorXd> map = const_vec.mapAt(0);
        EXPECT_EQ(map(0), 1); // Verify value
        EXPECT_EQ(map(1), 2);
    }

    // Test mapAt (writable)
    {
        Eigen::Map<Eigen::VectorXd> map = vec.mapAt(0);
        map(0) = 10;
        EXPECT_EQ(vec.at(0)(0), 10); // Verify modification
        map += Eigen::VectorXd::Ones(2); // Test += operation
        Eigen::VectorXd expected(2);
        expected << 11, 3;
        EXPECT_TRUE(vec.at(0).isApprox(expected));
    }

    // Test setFromScalar
    vec.setFromScalar(b_);
    PrintVector(vec, "After setFromScalar");
    EXPECT_TRUE(vec.toEigen().isApprox(b_));

    // Test toEigen
    EXPECT_TRUE(vec.toEigen().isApprox(b_));
}

// Test invalid inputs
TEST_F(BlockSparseMatrixVectorTest, InvalidInputs) {
    std::cout << "\n=== Testing Invalid Inputs ===\n";
    BlockSparseMatrix mat(block_sizes_, true);
    BlockVector vec(block_sizes_);

    // Matrix invalid index
    EXPECT_THROW(mat.add(3, 0, m_diag_2x2_), std::invalid_argument);
    EXPECT_THROW(mat.at(0, 3), std::invalid_argument);
    EXPECT_THROW(mat.rowEntryAt(3, 0, false), std::invalid_argument);

    // Matrix invalid size
    Eigen::MatrixXd wrong_size(2, 3);
    EXPECT_THROW(mat.add(0, 0, wrong_size), std::invalid_argument);

    // Symmetric matrix lower triangle
    EXPECT_THROW(mat.at(1, 0), std::invalid_argument);
    EXPECT_NO_THROW(mat.add(1, 0, m_zero_2x3_)); // Should warn and ignore

    // Vector invalid index
    EXPECT_THROW(vec.add(3, Eigen::VectorXd(2)), std::invalid_argument);
    EXPECT_THROW(vec.at(3), std::invalid_argument);
    EXPECT_THROW(vec.mapAt(3), std::invalid_argument);

    // Vector invalid size
    Eigen::VectorXd wrong_size_vec(4);
    EXPECT_THROW(vec.add(0, wrong_size_vec), std::invalid_argument);

    // setFromScalar invalid size
    Eigen::VectorXd wrong_size_scalar(5);
    EXPECT_THROW(vec.setFromScalar(wrong_size_scalar), std::invalid_argument);
}

// Test sparsity patterns and solving
TEST_F(BlockSparseMatrixVectorTest, SparsityAndSolve) {
    std::cout << "\n=== Testing Sparsity Patterns and Solving ===\n";

    // Setup tridiagonal matrix
    BlockSparseMatrix tri(block_sizes_, true);
    tri.add(0, 0, m_diag_2x2_);
    tri.add(0, 1, m_offdiag_2x3_);
    tri.add(1, 1, m_diag_3x3_);
    tri.add(1, 2, m_zero_3x1_);
    tri.add(2, 2, Eigen::MatrixXd::Identity(1, 1));
    PrintMatrixPattern(tri.toEigen(true), "Tridiagonal Matrix");

    // Setup block diagonal matrix
    BlockSparseMatrix blkdiag(block_sizes_, true);
    blkdiag.add(0, 0, m_diag_2x2_);
    blkdiag.add(1, 1, m_diag_3x3_);
    blkdiag.add(2, 2, Eigen::MatrixXd::Identity(1, 1));
    PrintMatrixPattern(blkdiag.toEigen(true), "Block Diagonal Matrix");

    // Test sub-block sparsity
    auto eig_tri_sparse = tri.toEigen(true);
    std::cout << "Tridiagonal non-zeros (sub-block): " << eig_tri_sparse.nonZeros() << "\n";
    EXPECT_EQ(eig_tri_sparse.nonZeros(), 14); // 4 + 1 + 7 + 0 + 1 + 1

    auto eig_tri_blk = tri.toEigen(false);
    std::cout << "Tridiagonal non-zeros (block-level): " << eig_tri_blk.nonZeros() << "\n";
    EXPECT_EQ(eig_tri_blk.nonZeros(), 32); // 4 + 6 + 9 + 3 + 1 + 6 + 3

    // Solve system
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.analyzePattern(eig_tri_sparse);
    solver.factorize(eig_tri_sparse);
    ASSERT_EQ(solver.info(), Eigen::Success);
    Eigen::VectorXd x1 = solver.solve(b_);
    std::cout << "Solution x1: " << x1.transpose() << "\n";

    // Reuse pattern
    solver.analyzePattern(eig_tri_blk);
    solver.factorize(eig_tri_blk);
    ASSERT_EQ(solver.info(), Eigen::Success);
    Eigen::VectorXd x2 = solver.solve(b_);
    std::cout << "Solution x2: " << x2.transpose() << "\n";
    EXPECT_LT((x1 - x2).norm(), 1e-6);
}

// Stress test with large matrix and vector
TEST_F(BlockSparseMatrixVectorTest, StressTest) {
    std::cout << "\n=== Stress Test ===\n";
    std::vector<unsigned int> large_blocks(1000, 10); // 1000 blocks of size 10x10
    BlockSparseMatrix mat(large_blocks, false); // Non-symmetric for simplicity
    BlockVector vec(large_blocks);

    // Measure add performance
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < 1000; ++i) {
        if (i < 999) { // Create banded structure
            mat.add(i, i, RandomSPDMatrix(10));
            mat.add(i, i + 1, 0.1 * RandomMatrix(10, 10));
        }
        vec.add(i, Eigen::VectorXd::Random(10));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time to add 1999 matrix blocks and 1000 vector blocks: " << duration << " us\n";

    // Measure toEigen performance
    start = std::chrono::high_resolution_clock::now();
    auto eig_mat = mat.toEigen(true);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time to convert to Eigen sparse matrix: " << duration << " us\n";
    std::cout << "Non-zeros: " << eig_mat.nonZeros() << "\n";

    // Measure matrix-vector multiplication performance
    Eigen::VectorXd vec_eigen = vec.toEigen();
    start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd result = eig_mat * vec_eigen;
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time for matrix-vector multiplication: " << duration << " us\n";

    // Verify correctness
    EXPECT_TRUE(vec.toEigen().size() == 10000); // 1000 * 10
    EXPECT_TRUE(mat.at(0, 0).rows() == 10 && mat.at(0, 0).cols() == 10);
    EXPECT_TRUE(result.size() == 10000); // Result size matches matrix rows
}

} // namespace finalicp

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}