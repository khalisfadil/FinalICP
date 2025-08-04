#include <solver/covariance.hpp>

#include <matrixoperator/matrix.hpp>
#include <matrixoperator/matrixsparse.hpp>
#include <matrixoperator/vector.hpp>

namespace finalicp {

    // ###########################################################
    // Covariance
    // ###########################################################

    Covariance::Covariance(Problem& problem)
            : state_vector_(problem.getStateVector()),
            hessian_solver_(std::make_shared<SolverType>()) {
#ifdef DEBUG
        std::cout << "[Covariance DEBUG] Constructing from Problem. Building and factorizing Hessian..." << std::endl;
#endif
        Eigen::SparseMatrix<double> approx_hessian;
        Eigen::VectorXd gradient_vector;
        problem.buildGaussNewtonTerms(approx_hessian, gradient_vector);
#ifdef DEBUG
        if (!approx_hessian.coeffs().allFinite()) {
            std::cerr << "[Covariance DEBUG] CRITICAL: Hessian built from problem contains non-finite values!" << std::endl;
        }
#endif
        hessian_solver_->analyzePattern(approx_hessian);
        hessian_solver_->factorize(approx_hessian);
        if (hessian_solver_->info() != Eigen::Success) {
            std::string error_msg = "[Covariance] Eigen LLT decomposition failed. ";
#ifdef DEBUG
            switch(hessian_solver_->info()) {
                case Eigen::NumericalIssue:
                    error_msg += "Reason: Numerical issue. The matrix is likely not positive-definite or is ill-conditioned.";
                    break;
                case Eigen::NoConvergence:
                    error_msg += "Reason: No convergence.";
                    break;
                case Eigen::InvalidInput:
                    error_msg += "Reason: Invalid input.";
                    break;
                default:
                    error_msg += "Reason: Unknown.";
                    break;
            }
#endif
             throw std::runtime_error(error_msg);
        }
#ifdef DEBUG
        std::cout << "    - Hessian factorization successful." << std::endl;
#endif
    }

    // ###########################################################
    // Covariance
    // ###########################################################

    Covariance::Covariance(GaussNewtonSolver& solver)
    : state_vector_(solver.state_vector()), hessian_solver_(solver.solver()) {
        if (hessian_solver_->info() != Eigen::Success) {
            throw std::runtime_error("[Covariance] Constructed with a solver that was not successfully factorized!");
        }
#ifdef DEBUG
        std::cout << "[Covariance DEBUG] Constructing from existing solver. Reusing factorization." << std::endl;
#endif
    }

    // ###########################################################
    // query
    // ###########################################################

    Eigen::MatrixXd Covariance::query(const StateVarBase::ConstPtr& var) const {
        return query(std::vector<StateVarBase::ConstPtr>{var});
    }

    // ###########################################################
    // query
    // ###########################################################

    Eigen::MatrixXd Covariance::query(const StateVarBase::ConstPtr& rvar, const StateVarBase::ConstPtr& cvar) const {
        return query(std::vector<StateVarBase::ConstPtr>{rvar},
                std::vector<StateVarBase::ConstPtr>{cvar});
    }

    // ###########################################################
    // query
    // ###########################################################

    Eigen::MatrixXd Covariance::query(const std::vector<StateVarBase::ConstPtr>& vars) const {
        return query(vars, vars);
    }

    // ###########################################################
    // query
    // ###########################################################

    Eigen::MatrixXd Covariance::query(const std::vector<StateVarBase::ConstPtr>& rvars, const std::vector<StateVarBase::ConstPtr>& cvars) const {
        const auto state_vector = state_vector_.lock();
        if (!state_vector) throw std::runtime_error{"state vector expired."};

        // Creating indexing
        BlockMatrixIndexing indexing(state_vector->getStateBlockSizes());
        const auto& blk_row_indexing = indexing.rowIndexing();
        const auto& blk_col_indexing = indexing.colIndexing();

        // Fixed sizes
        const auto num_row_vars = rvars.size();
        const auto num_col_vars = cvars.size();

#ifdef DEBUG
        // --- [IMPROVEMENT] Log which variables are being queried ---
        std::stringstream ss;
        ss << "    - Querying covariance. Rows: { ";
        for(const auto& var : rvars) { ss << var->key() << " "; }
        ss << "}, Cols: { ";
        for(const auto& var : cvars) { ss << var->key() << " "; }
        ss << "}";
        std::cout << ss.str() << std::endl;
#endif

        // Look up block indexes
        std::vector<unsigned int> blk_row_indices;
        blk_row_indices.reserve(num_row_vars);
        for (size_t i = 0; i < num_row_vars; i++)
            blk_row_indices.emplace_back(
                state_vector->getStateBlockIndex(rvars[i]->key()));

        std::vector<unsigned int> blk_col_indices;
        blk_col_indices.reserve(num_col_vars);
        for (size_t i = 0; i < num_col_vars; i++)
            blk_col_indices.emplace_back(
                state_vector->getStateBlockIndex(cvars[i]->key()));

        // Look up block size of state variables
        std::vector<unsigned int> blk_row_sizes;
        blk_row_sizes.reserve(num_row_vars);
        for (size_t i = 0; i < num_row_vars; i++)
            blk_row_sizes.emplace_back(blk_row_indexing.blkSizeAt(blk_row_indices[i]));

        std::vector<unsigned int> blk_col_sizes;
        blk_col_sizes.reserve(num_col_vars);
        for (size_t i = 0; i < num_col_vars; i++)
            blk_col_sizes.emplace_back(blk_col_indexing.blkSizeAt(blk_col_indices[i]));

        // Create result container
        BlockMatrix cov_blk(blk_row_sizes, blk_col_sizes);
        const auto& cov_blk_indexing = cov_blk.getIndexing();
        const auto& cov_blk_row_indexing = cov_blk_indexing.rowIndexing();
        const auto& cov_blk_col_indexing = cov_blk_indexing.colIndexing();

        // For each column key
        for (unsigned int c = 0; c < num_col_vars; c++) {
            // For each scalar column
            Eigen::VectorXd projection(blk_row_indexing.scalarSize());
            projection.setZero();
            for (unsigned int j = 0; j < blk_col_sizes[c]; j++) {
                // Get scalar index
                unsigned int scalar_col_index =
                    blk_col_indexing.cumSumAt(blk_col_indices[c]) + j;

                // Solve for scalar column of covariance matrix
                projection(scalar_col_index) = 1.0;
                Eigen::VectorXd x = hessian_solver_->solve(projection);
                projection(scalar_col_index) = 0.0;
#ifdef DEBUG
                // --- [IMPROVEMENT] Check the result of the linear solve ---
                // Log only for the very first column to avoid spam
                if (c == 0 && j == 0) {
                    std::cout << "      - Solving for first column of covariance block..." << std::endl;
                    if (!x.allFinite()) {
                        std::cerr << "      CRITICAL: Result of Hx=b solve is non-finite! Hessian is likely ill-conditioned." << std::endl;
                    } else {
                        std::cout << "      - Solution norm: " << x.norm() << std::endl;
                    }
                }
#endif

                // For each block row
                for (unsigned int r = 0; r < num_row_vars; r++) {
                    // Get scalar index into solution vector
                    int scalarRowIndex = blk_row_indexing.cumSumAt(blk_row_indices[r]);

                    // Do the backward pass, using the Cholesky factorization (fast)
                    cov_blk.at(r, c).block(0, j, blk_row_sizes[r], 1) =
                        x.block(scalarRowIndex, 0, blk_row_sizes[r], 1);
                }
            }
        }

        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(cov_blk_row_indexing.scalarSize(), cov_blk_col_indexing.scalarSize());

        for (unsigned int r = 0; r < num_row_vars; r++) {
            for (unsigned int c = 0; c < num_col_vars; c++) {
            cov.block(cov_blk_row_indexing.cumSumAt(r),
                        cov_blk_col_indexing.cumSumAt(c), blk_row_sizes[r],
                        blk_col_sizes[c]) = cov_blk.at(r, c);
            }
        }

#ifdef DEBUG
        if (!cov.allFinite()) {
            std::cerr << "[Covariance DEBUG] CRITICAL: Final assembled covariance block contains non-finite values!" << std::endl;
        }
#endif
        return cov;
    }
} // namespace finaleicp