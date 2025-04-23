#include <problem/optimizationproblem.hpp>

#include <iomanip>
#include <iostream>

namespace finalicp {
    auto OptimizationProblem::MakeShared(unsigned int num_threads)-> OptimizationProblem::Ptr {
        return std::make_shared<OptimizationProblem>(num_threads);
    }

    OptimizationProblem::OptimizationProblem(unsigned int num_threads)
    : num_threads_(num_threads) {}

    void OptimizationProblem::addStateVariable(const StateVarBase::Ptr &state) {
        state_vars_.push_back(state);
    }

    void OptimizationProblem::addCostTerm(const BaseCostTerm::ConstPtr &costTerm) {
        cost_terms_.push_back(costTerm);
    }

    unsigned int OptimizationProblem::getNumberOfCostTerms() const {
        return cost_terms_.size();
    }

    double OptimizationProblem::cost() const {
        // Initialize accumulators (thread-safe for parallel case)
        std::atomic<double> total_cost{0.0};
        std::atomic<size_t> nan_count{0};
        std::atomic<size_t> exception_count{0};

        // Sequential processing for small cost_terms_ to avoid parallel overhead
        if (cost_terms_.size() < 100) { // Tune threshold via profiling
            double cost = 0;
            for (size_t i = 0; i < cost_terms_.size(); i++) {
                try {
                    double cost_i = cost_terms_.at(i)->cost();
                    if (std::isnan(cost_i)) {
                        ++nan_count;
                    } else {
                        cost += cost_i;
                    }
                } catch (const std::exception& e) {
                    ++exception_count;
                    std::cerr << "[OptimizationProblem::cost] exception in cost term: " << e.what() << std::endl;
                } catch (...) {
                    ++exception_count;
                    std::cerr << "[OptimizationProblem::cost] exception in cost term: (unknown)" << std::endl;
                }
            }
            if (nan_count > 0) {
                std::cerr << "[OptimizationProblem::cost] Warning: " << nan_count << " NaN cost terms ignored!" << std::endl;
            }
            if (exception_count > 0) {
                std::cerr << "[OptimizationProblem::cost] Warning: " << exception_count << " exceptions occurred in cost terms!" << std::endl;
            }
            return cost;
        }

        // Parallel processing with TBB parallel_for for large cost_terms_
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads_);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cost_terms_.size(), 100),
            [&total_cost, &nan_count, &exception_count, this](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    try {
                        double cost_i = cost_terms_.at(i)->cost();
                        if (std::isnan(cost_i)) {
                            ++nan_count; // Atomic increment
                        } else {
                            total_cost += cost_i; // Atomic addition
                        }
                    } catch (const std::exception& e) {
                        ++exception_count; // Atomic increment
                        std::cerr << "[OptimizationProblem::cost] exception in cost term at index " << i << ": " << e.what() << std::endl;
                    } catch (...) {
                        ++exception_count; // Atomic increment
                        std::cerr << "[OptimizationProblem::cost] exception in cost term at index " << i << ": (unknown)" << std::endl;
                    }
                }
            }
        );

        // Log warnings after parallel processing
        if (nan_count > 0) {
            std::cerr << "[OptimizationProblem::cost] Warning: " << nan_count << " NaN cost terms ignored!" << std::endl;
        }
        if (exception_count > 0) {
            std::cerr << "[OptimizationProblem::cost] Warning: " << exception_count << " exceptions occurred in cost terms!" << std::endl;
        }

        return total_cost;
    }

    StateVector::Ptr OptimizationProblem::getStateVector() const {
    *state_vector_ = StateVector();
        for (const auto &state_var : state_vars_) {
            if (!state_var->locked()) state_vector_->addStateVariable(state_var);
        }
        return state_vector_;
    }

    void OptimizationProblem::buildGaussNewtonTerms(
        Eigen::SparseMatrix<double>& approximate_hessian,
        Eigen::VectorXd& gradient_vector) const {
        // Initialize block matrices
        std::vector<unsigned int> sqSizes = state_vector_->getStateBlockSizes();
        BlockSparseMatrix A_(sqSizes, true);
        BlockVector b_(sqSizes);

        // Track exceptions for thread-safe logging
        std::atomic<size_t> exception_count{0};

        // Process cost terms: sequential for small sizes, parallel for large
        if (cost_terms_.size() < 100) { // Tune threshold via profiling
            for (unsigned int c = 0; c < cost_terms_.size(); c++) {
                try {
                    cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
                } catch (const std::exception& e) {
                    ++exception_count;
                    std::cerr << "[OptimizationProblem::buildGaussNewtonTerms] exception in cost term: " << e.what() << std::endl;
                } catch (...) {
                    ++exception_count;
                    std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  exception in cost term: (unknown)" << std::endl;
                }
            }
        } else {
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads_);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cost_terms_.size(), 100),
                [&A_, &b_, &exception_count, this](const tbb::blocked_range<size_t>& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        try {
                            cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
                        } catch (const std::exception& e) {
                            ++exception_count; // Atomic increment
                            std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  exception in cost term at index " << c << ": " << e.what() << std::endl;
                        } catch (...) {
                            ++exception_count; // Atomic increment
                            std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  exception in cost term at index " << c << ": (unknown)" << std::endl;
                        }
                    }
                }
            );
        }

        // Log exceptions after processing
        if (exception_count > 0) {
            std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  Warning: " << exception_count << " exceptions occurred in cost terms!" << std::endl;
        }

        // Convert to Eigen types with block-sparsity pattern
        // Note: Sub-block sparsity is not exploited, as it may change in later iterations
        approximate_hessian = A_.toEigen(false);
        gradient_vector = b_.toEigen();
    }



} // namespace finalicp