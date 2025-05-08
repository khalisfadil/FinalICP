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

        // Sequential processing for small cost_terms_ to avoid parallel overhead
        // if (cost_terms_.size() < 100) { // Tune threshold via profiling
            double cost = 0;
            for (size_t i = 0; i < cost_terms_.size(); i++) {
                try {
                    double cost_i = cost_terms_.at(i)->cost();
                    if (std::isnan(cost_i)) {
                        std::cerr << "[OptimizationProblem::cost] NaN cost term is ignored! " << std::endl;
                    } else {
                        cost += cost_i;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[OptimizationProblem::cost] exception in cost term: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[OptimizationProblem::cost] exception in cost term: (unknown)" << std::endl;
                }
            }
            return cost;
        // }

        // // Parallel processing with TBB parallel_reduce for large cost_terms_
        // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads_);
        // double total_cost = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, cost_terms_.size(), 100), 0.0, 
        //     [this](const tbb::blocked_range<size_t>& range, double init) -> double {
        //         double local_cost = init;
        //         for (size_t i = range.begin(); i != range.end(); ++i) {
        //             try {
        //                 double cost_i = cost_terms_.at(i)->cost();
        //                 if (std::isnan(cost_i)) {
        //                     std::cerr << "[OptimizationProblem::cost] NaN cost term is ignored! " << std::endl;
        //                 } else {
        //                     local_cost += cost_i;
        //                 }
        //             } catch (const std::exception& e) {
            
        //                 std::cerr << "[OptimizationProblem::cost] exception in cost term at index " << i << ": " << e.what() << std::endl;
        //             } catch (...) {
                        
        //                 std::cerr << "[OptimizationProblem::cost] exception in cost term at index " << i << ": (unknown)" << std::endl;
        //             }
        //         }
        //         return local_cost; // Returns partial sum to TBB, not exiting cost
        //     },
        //     std::plus<double>() // Combine partial results
        // );
        // return total_cost;
    }

    StateVector::Ptr OptimizationProblem::getStateVector() const {
        *state_vector_ = StateVector();
        for (const auto &state_var : state_vars_) {
            if (!state_var->locked()) state_vector_->addStateVariable(state_var);

            //debug
            // ##################################
            std::cout << "Adding state var with key: " << state_var->key() << ", perturb_dim: " << state_var->perturb_dim() << std::endl;
            // ##################################
        }

        //debug
        // ##################################
        std::cout << "State vector block sizes: ";
        for (const auto& size : state_vector_->getStateBlockSizes()) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
        // ##################################

        return state_vector_;
    }

    void OptimizationProblem::buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian, Eigen::VectorXd& gradient_vector) const {
        
        std::vector<unsigned int> sqSizes = state_vector_->getStateBlockSizes();
        
        // debug
        // ##################################
        std::cout << "buildGaussNewtonTerms - sqSizes: ";
        for (const auto& size : sqSizes) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
        // ##################################
        
        BlockSparseMatrix A_(sqSizes, true);
        BlockVector b_(sqSizes);

        // debug
        // ##################################
        std::cout << "Gradient vector num entries: " << b_.getIndexing().numEntries() << std::endl;
        // ##################################

        // Process cost terms: sequential for small sizes, parallel for large
        if (cost_terms_.size() < 100) { // Tune threshold via profiling
            for (unsigned int c = 0; c < cost_terms_.size(); c++) {
                try {

                    cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
                } catch (const std::exception& e) {
                    std::cerr << "[OptimizationProblem::buildGaussNewtonTerms] exception in cost term: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  exception in cost term: (unknown)" << std::endl;
                }
            }
        } 
        else {
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads_);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cost_terms_.size(), 100),
                [&A_, &b_, this](const tbb::blocked_range<size_t>& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        try {
                            cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
                        } catch (const std::exception& e) {
                            std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  exception in cost term at index " << c << ": " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  exception in cost term at index " << c << ": (unknown)" << std::endl;
                        }
                    }
                }
            );
        }

        // debug
        // ##################################
        std::cout << "Converting BlockSparseMatrix to Eigen" << std::endl;
        // ##################################

        approximate_hessian = A_.toEigen(false);

        // debug
        // ##################################
        std::cout << "Converting BlockVector to Eigen, size: " << b_.toEigen().size() << std::endl;
        // ##################################

        gradient_vector = b_.toEigen();
    }
} // namespace finalicp