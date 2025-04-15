#include <gtest/gtest.h>

#include <iostream>

#include <matrixoperator/matrixsparse.hpp>
#include <matrixoperator/vector.hpp>
#include <matrixoperator/matrix.hpp>

TEST(finalicp, SparsityPattern) {
  std::vector<unsigned int> blockSizes;
  blockSizes.resize(3);
  blockSizes[0] = 2;
  blockSizes[1] = 2;
  blockSizes[2] = 2;

  // Setup blocks
  Eigen::Matrix2d m_tri_diag;
  m_tri_diag << 2, 1, 1, 2;
  Eigen::Matrix2d m_tri_offdiag;
  m_tri_offdiag << 0, 0, 1, 0;
  Eigen::Matrix2d m_z = Eigen::Matrix2d::Zero();
  Eigen::Matrix2d m_o = Eigen::Matrix2d::Ones();

  // Setup A - Case 1
  finalicp::BlockSparseMatrix tri(blockSizes, true);
  tri.add(0, 0, m_tri_diag);
  tri.add(0, 1, m_tri_offdiag);
  tri.add(1, 1, m_tri_diag);
  tri.add(1, 2, m_tri_offdiag);
  tri.add(2, 2, m_tri_diag);

  // Setup A - Case 2
  finalicp::BlockSparseMatrix blkdiag(blockSizes, true);
  blkdiag.add(0, 0, m_tri_diag);
  blkdiag.add(0, 1, m_z);
  blkdiag.add(1, 1, m_tri_diag);
  blkdiag.add(1, 2, m_z);
  blkdiag.add(2, 2, m_tri_diag);

  // Setup A - Case 3
  finalicp::BlockSparseMatrix tri_ones(blockSizes, true);
  tri_ones.add(0, 0, m_o);
  tri_ones.add(0, 1, m_o);
  tri_ones.add(1, 1, m_o);
  tri_ones.add(1, 2, m_o);
  tri_ones.add(2, 2, m_o);

  // Setup B
  Eigen::VectorXd b(6);
  b << 1, 2, 3, 4, 5, 6;

  // Test sub sparsity
  {
    // Maximum sparsity
    Eigen::SparseMatrix<double> eig_sparse = tri.toEigen(true);
    std::cout << "case1: " << eig_sparse << std::endl;
    std::cout << "nonzeros: " << eig_sparse.nonZeros() << std::endl;
    EXPECT_EQ(eig_sparse.nonZeros(), 14);

    // Get only block-level sparsity (important for re-using pattern)
    Eigen::SparseMatrix<double> eig_blk_sparse = tri.toEigen(false);
    std::cout << "case2: " << eig_blk_sparse << std::endl;
    std::cout << "nonzeros: " << eig_blk_sparse.nonZeros() << std::endl;
    EXPECT_EQ(eig_blk_sparse.nonZeros(), 20);
  }

  // Test solve
  {
    // Maximum sparsity
    Eigen::SparseMatrix<double> eig_sparse = tri.toEigen(true);
    std::cout << "case1: " << eig_sparse << std::endl;
    std::cout << "nonzeros: " << eig_sparse.nonZeros() << std::endl;

    // Solve sparse
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.analyzePattern(eig_sparse);
    solver.factorize(eig_sparse);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("Decomp failure.");
    Eigen::VectorXd x1 = solver.solve(b);
    std::cout << "x1: " << x1.transpose() << std::endl;

    // Get only block-level sparsity (important for re-using pattern)
    Eigen::SparseMatrix<double> eig_blk_sparse = tri.toEigen(false);
    std::cout << "case2: " << eig_blk_sparse << std::endl;
    std::cout << "nonzeros: " << eig_blk_sparse.nonZeros() << std::endl;

    // Solve sparse
    solver.analyzePattern(eig_blk_sparse);
    solver.factorize(eig_blk_sparse);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error("Decomp failure.");
    Eigen::VectorXd x2 = solver.solve(b);
    std::cout << "x2: " << x2.transpose() << std::endl;
    EXPECT_LT((x1 - x2).norm(), 1e-6);
  }

  // Test solve, setting pattern with ones
  {
    // Solve using regular tri-block diagonal
    Eigen::SparseMatrix<double> eig_tri = tri.toEigen();
    std::cout << "case1: " << eig_tri << std::endl;
    std::cout << "nonzeros: " << eig_tri.nonZeros() << std::endl;

    // Solve sparse
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.analyzePattern(eig_tri);
    solver.factorize(eig_tri);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x1 = solver.solve(b);
    std::cout << "x1: " << x1.transpose() << std::endl;

    // Set pattern using ones and then solve with tri-block
    Eigen::SparseMatrix<double> eig_tri_ones = tri_ones.toEigen();
    std::cout << "case2: " << eig_tri_ones << std::endl;
    std::cout << "nonzeros: " << eig_tri_ones.nonZeros() << std::endl;

    // Solve sparse
    solver.analyzePattern(eig_tri_ones);
    solver.factorize(eig_tri);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x2 = solver.solve(b);
    std::cout << "x2: " << x2.transpose() << std::endl;
    EXPECT_LT((x1 - x2).norm(), 1e-6);
  }

  // Test solve of matrix with zero blocks, setting pattern with ones
  {
    // Solve using regular tri-block diagonal
    Eigen::SparseMatrix<double> eig_blkdiag = blkdiag.toEigen();
    std::cout << "case1: " << eig_blkdiag << std::endl;
    std::cout << "nonzeros: " << eig_blkdiag.nonZeros() << std::endl;

    // Solve sparse
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.analyzePattern(eig_blkdiag);
    solver.factorize(eig_blkdiag);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x1 = solver.solve(b);
    std::cout << "x1: " << x1.transpose() << std::endl;

    // Set pattern using ones and then solve with tri-block
    Eigen::SparseMatrix<double> eig_tri_ones = tri_ones.toEigen();
    std::cout << "case2: " << eig_tri_ones << std::endl;
    std::cout << "nonzeros: " << eig_tri_ones.nonZeros() << std::endl;

    // Solve sparse
    solver.analyzePattern(eig_tri_ones);
    solver.factorize(eig_blkdiag);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x2 = solver.solve(b);
    std::cout << "x2: " << x2.transpose() << std::endl;
    EXPECT_LT((x1 - x2).norm(), 1e-6);
  }
}

TEST(finalicp, BlockVectorTest) {
  // Define block sizes
  std::vector<unsigned int> blockSizes = {2, 3, 1};

  // Test default constructor
  {
    finalicp::BlockVector vec;
    EXPECT_EQ(vec.toEigen().size(), 0);
  }

  // Test block size constructor
  {
    finalicp::BlockVector vec(blockSizes);
    EXPECT_EQ(vec.toEigen().size(), 6); // 2 + 3 + 1
    EXPECT_TRUE(vec.toEigen().isZero());
    EXPECT_EQ(vec.getIndexing().numEntries(), 3);
    EXPECT_EQ(vec.getIndexing().blkSizeAt(0), 2);
    EXPECT_EQ(vec.getIndexing().blkSizeAt(1), 3);
    EXPECT_EQ(vec.getIndexing().blkSizeAt(2), 1);
  }

  // Test block size with data constructor
  {
    Eigen::VectorXd data(6);
    data << 1, 2, 3, 4, 5, 6;
    finalicp::BlockVector vec(blockSizes, data);
    EXPECT_EQ(vec.toEigen().size(), 6);
    EXPECT_TRUE(vec.toEigen().isApprox(data));
  }

  // Test invalid data size in constructor
  {
    Eigen::VectorXd data(5); // Wrong size (expects 6)
    data << 1, 2, 3, 4, 5;
    EXPECT_THROW(finalicp::BlockVector(blockSizes, data), std::invalid_argument);
  }

  // Test setFromScalar
  {
    finalicp::BlockVector vec(blockSizes);
    Eigen::VectorXd data(6);
    data << 1, 2, 3, 4, 5, 6;
    vec.setFromScalar(data);
    EXPECT_TRUE(vec.toEigen().isApprox(data));
  }

  // Test setFromScalar with invalid size
  {
    finalicp::BlockVector vec(blockSizes);
    Eigen::VectorXd data(5); // Wrong size
    EXPECT_THROW(vec.setFromScalar(data), std::invalid_argument);
  }

  // Test add and at
  {
    finalicp::BlockVector vec(blockSizes);
    Eigen::VectorXd v0(2);
    v0 << 1, 2;
    Eigen::VectorXd v1(3);
    v1 << 3, 4, 5;
    Eigen::VectorXd v2(1);
    v2 << 6;

    vec.add(0, v0);
    vec.add(1, v1);
    vec.add(2, v2);

    EXPECT_TRUE(vec.at(0).isApprox(v0));
    EXPECT_TRUE(vec.at(1).isApprox(v1));
    EXPECT_TRUE(vec.at(2).isApprox(v2));

    Eigen::VectorXd expected(6);
    expected << 1, 2, 3, 4, 5, 6;
    EXPECT_TRUE(vec.toEigen().isApprox(expected));
  }

  // Test add with invalid block index
  {
    finalicp::BlockVector vec(blockSizes);
    Eigen::VectorXd v(2);
    EXPECT_THROW(vec.add(3, v), std::invalid_argument);
  }

  // Test add with invalid vector size
  {
    finalicp::BlockVector vec(blockSizes);
    Eigen::VectorXd v(1); // Wrong size for block 0 (expects 2)
    EXPECT_THROW(vec.add(0, v), std::invalid_argument);
  }

  // Test at with invalid block index
  {
    finalicp::BlockVector vec(blockSizes);
    EXPECT_THROW(vec.at(3), std::invalid_argument);
  }

  // Test mapAt
  {
    finalicp::BlockVector vec(blockSizes);
    Eigen::VectorXd v0(2);
    v0 << 1, 2;
    vec.add(0, v0);

    Eigen::Map<Eigen::VectorXd> map = vec.mapAt(0);
    EXPECT_EQ(map.size(), 2);
    EXPECT_TRUE(map.isApprox(v0));

    // Modify through map
    map(0) = 10;
    EXPECT_EQ(vec.at(0)(0), 10);
  }

  // Test mapAt with invalid block index
  {
    finalicp::BlockVector vec(blockSizes);
    EXPECT_THROW(vec.mapAt(3), std::invalid_argument);
  }

  // Test cumulative addition
  {
    finalicp::BlockVector vec(blockSizes);
    Eigen::VectorXd v0(2);
    v0 << 1, 2;
    vec.add(0, v0);
    vec.add(0, v0); // Add again
    Eigen::VectorXd expected(2);
    expected << 2, 4;
    EXPECT_TRUE(vec.at(0).isApprox(expected));
  }
}

TEST(finalicp, BlockMatrixTest) {
  // Define block sizes
  std::vector<unsigned int> blockSizes = {2, 3, 1};
  std::vector<unsigned int> rectRowSizes = {2, 3};
  std::vector<unsigned int> rectColSizes = {3, 1};

  // Test default constructor
  {
    finalicp::BlockMatrix mat;
    // Default constructor doesn't initialize data_, so we can't test much beyond creation
    EXPECT_NO_THROW(mat.zero()); // Should not crash
  }

  // Test rectangular matrix constructor
  {
    finalicp::BlockMatrix mat(rectRowSizes, rectColSizes);
    EXPECT_EQ(mat.getIndexing().rowIndexing().numEntries(), 2);
    EXPECT_EQ(mat.getIndexing().colIndexing().numEntries(), 2);
    EXPECT_EQ(mat.getIndexing().rowIndexing().blkSizeAt(0), 2);
    EXPECT_EQ(mat.getIndexing().rowIndexing().blkSizeAt(1), 3);
    EXPECT_EQ(mat.getIndexing().colIndexing().blkSizeAt(0), 3);
    EXPECT_EQ(mat.getIndexing().colIndexing().blkSizeAt(1), 1);
    EXPECT_TRUE(mat.copyAt(0, 0).isZero());
    EXPECT_EQ(mat.copyAt(0, 0).rows(), 2);
    EXPECT_EQ(mat.copyAt(0, 0).cols(), 3);
    EXPECT_EQ(mat.copyAt(1, 1).rows(), 3);
    EXPECT_EQ(mat.copyAt(1, 1).cols(), 1);
  }

  // Test symmetric matrix constructor
  {
    finalicp::BlockMatrix mat(blockSizes, true);
    EXPECT_EQ(mat.getIndexing().rowIndexing().numEntries(), 3);
    EXPECT_EQ(mat.getIndexing().colIndexing().numEntries(), 3);
    EXPECT_EQ(mat.getIndexing().rowIndexing().blkSizeAt(0), 2);
    EXPECT_EQ(mat.getIndexing().rowIndexing().blkSizeAt(1), 3);
    EXPECT_EQ(mat.getIndexing().rowIndexing().blkSizeAt(2), 1);
    EXPECT_TRUE(mat.copyAt(0, 0).isZero());
    EXPECT_EQ(mat.copyAt(0, 0).rows(), 2);
    EXPECT_EQ(mat.copyAt(0, 0).cols(), 2);
    EXPECT_EQ(mat.copyAt(1, 1).rows(), 3);
    EXPECT_EQ(mat.copyAt(1, 1).cols(), 3);
    EXPECT_EQ(mat.copyAt(2, 2).rows(), 1);
    EXPECT_EQ(mat.copyAt(2, 2).cols(), 1);
  }

  // Test zero
  {
    finalicp::BlockMatrix mat(blockSizes, false);
    Eigen::MatrixXd m(2, 2);
    m << 1, 2, 3, 4;
    mat.add(0, 0, m);
    mat.zero();
    EXPECT_TRUE(mat.copyAt(0, 0).isZero());
    EXPECT_TRUE(mat.copyAt(1, 1).isZero());
    EXPECT_TRUE(mat.copyAt(2, 2).isZero());
  }

  // Test add and at
  {
    finalicp::BlockMatrix mat(blockSizes, false);
    Eigen::MatrixXd m0(2, 2);
    m0 << 1, 2, 3, 4;
    Eigen::MatrixXd m1(3, 3);
    m1 << 5, 6, 7, 8, 9, 10, 11, 12, 13;
    Eigen::MatrixXd m2(1, 1);
    m2 << 14;

    mat.add(0, 0, m0);
    mat.add(1, 1, m1);
    mat.add(2, 2, m2);

    EXPECT_TRUE(mat.at(0, 0).isApprox(m0));
    EXPECT_TRUE(mat.at(1, 1).isApprox(m1));
    EXPECT_TRUE(mat.at(2, 2).isApprox(m2));
    EXPECT_TRUE(mat.copyAt(0, 0).isApprox(m0));
    EXPECT_TRUE(mat.copyAt(1, 1).isApprox(m1));
    EXPECT_TRUE(mat.copyAt(2, 2).isApprox(m2));
  }

  // Test cumulative add
  {
    finalicp::BlockMatrix mat(blockSizes, false);
    Eigen::MatrixXd m(2, 2);
    m << 1, 2, 3, 4;
    mat.add(0, 0, m);
    mat.add(0, 0, m); // Add again
    Eigen::MatrixXd expected(2, 2);
    expected << 2, 4, 6, 8;
    EXPECT_TRUE(mat.at(0, 0).isApprox(expected));
  }

  // Test add with invalid indices
  {
    finalicp::BlockMatrix mat(blockSizes, false);
    Eigen::MatrixXd m(2, 2);
    EXPECT_THROW(mat.add(3, 0, m), std::invalid_argument);
    EXPECT_THROW(mat.add(0, 3, m), std::invalid_argument);
  }

  // Test add with invalid matrix size
  {
    finalicp::BlockMatrix mat(blockSizes, false);
    Eigen::MatrixXd m(2, 3); // Wrong size for (0,0)
    EXPECT_THROW(mat.add(0, 0, m), std::invalid_argument);
  }

  // Test at with invalid indices
  {
    finalicp::BlockMatrix mat(blockSizes, false);
    EXPECT_THROW(mat.at(3, 0), std::invalid_argument);
    EXPECT_THROW(mat.at(0, 3), std::invalid_argument);
  }

  // Test copyAt with invalid indices
  {
    finalicp::BlockMatrix mat(blockSizes, false);
    EXPECT_THROW(mat.copyAt(3, 0), std::invalid_argument);
    EXPECT_THROW(mat.copyAt(0, 3), std::invalid_argument);
  }

  // Test symmetric matrix add (lower triangle ignored)
  {
    finalicp::BlockMatrix mat(blockSizes, true);
    Eigen::MatrixXd m(2, 3);
    m << 1, 2, 3, 4, 5, 6;
    mat.add(1, 0, m); // Lower triangle, should be ignored
    EXPECT_TRUE(mat.copyAt(1, 0).isZero());
    mat.add(0, 1, m); // Upper triangle
    EXPECT_TRUE(mat.copyAt(0, 1).isApprox(m));
    EXPECT_TRUE(mat.copyAt(1, 0).isApprox(m.transpose())); // Symmetric access
  }

  // Test symmetric matrix at (lower triangle throws)
  {
    finalicp::BlockMatrix mat(blockSizes, true);
    EXPECT_THROW(mat.at(1, 0), std::runtime_error);
  }

  // Test symmetric matrix copyAt
  {
    finalicp::BlockMatrix mat(blockSizes, true);
    Eigen::MatrixXd m(2, 3);
    m << 1, 2, 3, 4, 5, 6;
    mat.add(0, 1, m);
    EXPECT_TRUE(mat.copyAt(0, 1).isApprox(m));
    EXPECT_TRUE(mat.copyAt(1, 0).isApprox(m.transpose()));
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}