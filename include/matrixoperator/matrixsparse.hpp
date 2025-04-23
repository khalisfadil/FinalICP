#pragma once

#include <vector>
#include <map>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <matrixoperator/matrixbase.hpp>

namespace finalicp{
    
    //A thread-safe, block-sparse matrix optimized.
    class BlockSparseMatrix : public BlockMatrixBase {
        public:
            //Represents a single dense block in the matrix.
            class BlockRowEntry{
                public:
                    //constructor
                    BlockRowEntry() {}

                    //destructor
                    ~BlockRowEntry() {}

                    //Row entry
                    Eigen::MatrixXd data;
            };

            //Constructs an empty block-sparse matrix (size must be set).
            BlockSparseMatrix();

            //Constructs a rectangular block-sparse matrix.
            BlockSparseMatrix(const std::vector<unsigned int>& blkRowSizes,
                    const std::vector<unsigned int>& blkColSizes);

            //Constructs a symmetric block-sparse matrix.
            BlockSparseMatrix(const std::vector<unsigned int>& blkSizes, bool symmetric = false);

            //Clear sparse entries, maintain size
            void clear();

            //Keep the existing entries and sizes, but set them to zero
            virtual void zero();

            //Adds the matrix to the block entry at index (r,c), block dim must match
            virtual void add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m);

            //Accesses or inserts a block at (r, c).
            BlockRowEntry& rowEntryAt(unsigned int r, unsigned int c, bool allowInsert = false);

            //Returns a mutable reference to the block at (r, c).
            virtual Eigen::MatrixXd& at(unsigned int r, unsigned int c);

            //Returns a copy of the block at (r, c), or zero matrix if absent.
            virtual Eigen::MatrixXd copyAt(unsigned int r, unsigned int c) const;

            //Converts the block-sparse matrix to an Eigen sparse matrix
            Eigen::SparseMatrix<double> toEigen(bool getSubBlockSparsity = false) const;

            
        private:

            //Computes non-zero entries per scalar column.
            Eigen::VectorXi getNnzPerCol() const;  
            
            //Represents a sparse column with thread-safe row entries.
            class BlockSparseColumn {
                public:
                    //constructor
                    BlockSparseColumn() {}

                    //destructor
                    ~BlockSparseColumn() {}

                    //thread safe map
                    std::map<unsigned int, BlockRowEntry> rows;
            };

            //Column-wise storage of sparse blocks.
            std::vector<BlockSparseColumn> cols_;         
    };
} // namespace finalicp