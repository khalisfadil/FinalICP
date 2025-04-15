#pragma once

#include <vector>

#include <Eigen/Core>

#include <matrixoperator/matrixindexing.hpp>

namespace finalicp{
    //Base class for managing block matrix structures in factor graph optimization.
    class BlockMatrixBase{
        public:
            //Default constructor
            BlockMatrixBase();

            //Constructs a rectangular block matrix.
            BlockMatrixBase(const std::vector<unsigned int>& blkRowSizes,
                  const std::vector<unsigned int>& blkColSizes);

            //Block-size-symmetric matrix constructor, pure scalar symmetry is still optional
            BlockMatrixBase(const std::vector<unsigned int>& blkSqSizes, bool symmetric = false);

            //Interface for zero'ing all entries
            virtual void zero() = 0;

            //Get indexing object
            const BlockMatrixIndexing& getIndexing() const;

            //Get if matrix is symmetric on a scalar level
            bool isSymmetric() const;

            //Adds the matrix to the block entry at index (r,c), block dim must match
            virtual void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) = 0;

            //Accesses a reference to the matrix at (r, c).
            virtual Eigen::MatrixXd& at(unsigned int r, unsigned int c) = 0;

            //Returns a copy of the entry at index (r,c)
            virtual Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const = 0;

        private:

            bool symmetric_;                //True if the matrix is symmetric at the scalar level.
            BlockMatrixIndexing indexing_;  //Manages block-wise indexing.

    };
} // finalicp