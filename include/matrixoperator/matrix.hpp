#pragma once

#include <matrixoperator/matrixbase.hpp>

namespace finalicp{
    //Implements a dense block matrix for factor graph-based optimization.
    class BlockMatrix : public BlockMatrixBase {
        public:
            //Default constructor, matrix size must still be set before using
            BlockMatrix();

            //Rectangular matrix constructor
            BlockMatrix(const std::vector<unsigned int>& blkRowSizes, const std::vector<unsigned int>& blkColSizes);

            //Block-size-symmetric matrix constructor, pure scalar symmetry is still optional
            BlockMatrix(const std::vector<unsigned int>& blkSizes, bool symmetric = false);

            //Set entries to zero
            virtual void zero();

            //Adds the matrix to the block entry at index (r,c), block dim must match
            virtual void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m);

            //Returns a reference to the value at (r,c)
            virtual Eigen::MatrixXd& at(unsigned int r, unsigned int c);

            //Returns a copy of the entry at index (r,c)
            virtual Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const;

        private:

            //Matrix of block matrices (row, column)
            std::vector<std::vector<Eigen::MatrixXd> > data_;
    };
}   // finalicp