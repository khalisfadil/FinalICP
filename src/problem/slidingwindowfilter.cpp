#include <problem/slidingwindowfilter.hpp>

#include <iomanip>
#include <iostream>

namespace finalicp {
    // -----------------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------------

    auto SlidingWindowFilter::MakeShared(unsigned int num_threads)-> SlidingWindowFilter::Ptr {
        return std::make_shared<SlidingWindowFilter>(num_threads);
    }

    SlidingWindowFilter::SlidingWindowFilter(unsigned int num_threads)
        : num_threads_(num_threads) {}

    // -----------------------------------------------------------------------------
    // addStateVariable
    // -----------------------------------------------------------------------------

    void SlidingWindowFilter::addStateVariable(const StateVarBase::Ptr &variable) {
        addStateVariable(std::vector<StateVarBase::Ptr>{variable});
    }

    void SlidingWindowFilter::addStateVariable(const std::vector<StateVarBase::Ptr>& variables) {
        for (const auto& var : variables) {
            const auto res = variables_.try_emplace(var->key(), var, false);
            if (!res.second) throw std::runtime_error("duplicated variable key");
            variable_queue_.emplace_back(var->key());
            related_var_keys_.try_emplace(var->key(), KeySet{var->key()});
        }
    }

    // -----------------------------------------------------------------------------
    // marginalizeVariable
    // -----------------------------------------------------------------------------

    void SlidingWindowFilter::marginalizeVariable(const StateVarBase::Ptr &variable) {
        marginalizeVariable(std::vector<StateVarBase::Ptr>{variable});
    }

    void SlidingWindowFilter::marginalizeVariable(const std::vector<StateVarBase::Ptr>& variables) {
        if (variables.empty()) return;

        // Mark variables for marginalization in the persistent variables_ map
        for (const auto& var : variables) {
            variables_.at(var->key()).marginalize = true;
        }

        // Process variable queue and identify fixed variables
        StateVector fixed_state_vector;
        StateVector state_vector;

        std::vector<StateKey> to_remove;

        bool fixed = true;
        for (const auto& key : variable_queue_) {
            const auto& var = variables_.at(key);
            const auto& related_keys = related_var_keys_.at(key);
            if (std::all_of(related_keys.begin(), related_keys.end(),
                            [this](const StateKey& k) { return variables_.at(k).marginalize; })) {
                if (!fixed) throw std::runtime_error("[SlidingWindowFilter::marginalizeVariable] fixed variables must be at the front");
                fixed_state_vector.addStateVariable(var.variable);
                to_remove.emplace_back(key);
            } else {
                fixed = false;
            }
            state_vector.addStateVariable(var.variable);
        }

        // Process cost terms: sequential for small sizes, parallel for large
        tbb::concurrent_vector<BaseCostTerm::ConstPtr> active_cost_terms;
        active_cost_terms.reserve(cost_terms_.size()); // Pre-allocate for efficiency

        // Prepare matrices
        const auto state_sizes = state_vector.getStateBlockSizes();
        BlockSparseMatrix A_(state_sizes, true);
        BlockVector b_(state_sizes);

        if (cost_terms_.size() < 100) { // Sequential for small sizes
            for (unsigned int c = 0; c < cost_terms_.size(); c++) {
                KeySet keys;
                cost_terms_.at(c)->getRelatedVarKeys(keys);
                if (std::all_of(keys.begin(), keys.end(), [this](const StateKey& k) { 
                    return variables_.at(k).marginalize; })) {
                    cost_terms_.at(c)->buildGaussNewtonTerms(state_vector, &A_, &b_);
                } else {
                    active_cost_terms.push_back(cost_terms_.at(c));
                }
            }
        } else { // Parallel with TBB for large sizes
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads_);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cost_terms_.size(), 100),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        KeySet local_keys;
                        cost_terms_.at(c)->getRelatedVarKeys(local_keys); // Populate with StateKey values
                        if (std::all_of(local_keys.begin(), local_keys.end(),[this](const StateKey& k) { 
                            return variables_.at(k).marginalize; })) {
                            cost_terms_.at(c)->buildGaussNewtonTerms(state_vector, &A_, &b_);
                        } else {
                            active_cost_terms.push_back(cost_terms_.at(c));
                        }
                    }
                }
            );
        }

        // Update persistent cost_terms_ with active terms
        cost_terms_.assign(active_cost_terms.begin(), active_cost_terms.end());

        /// \todo use sparse matrix (e.g., Eigen::SparseMatrix<double> instead of dense Eigen::MatrixXd)
        Eigen::MatrixXd Aupper(A_.toEigen(false));
        Eigen::MatrixXd A(Aupper.selfadjointView<Eigen::Upper>());
        Eigen::VectorXd b(b_.toEigen());

        // Add cached terms
        if (fixed_A_.size() > 0) {
            A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
            b.head(fixed_b_.size()) += fixed_b_;
        }

        // Marginalize fixed variables (Schur complement)
        const auto fixed_state_size = fixed_state_vector.getStateSize();
        if (fixed_state_size > 0) {
            Eigen::MatrixXd A00(A.topLeftCorner(fixed_state_size, fixed_state_size));
            Eigen::MatrixXd A10(A.bottomLeftCorner(A.rows() - fixed_state_size, fixed_state_size));
            Eigen::MatrixXd A11(A.bottomRightCorner(A.rows() - fixed_state_size, A.cols() - fixed_state_size));
            Eigen::VectorXd b0(b.head(fixed_state_size));
            Eigen::VectorXd b1(b.tail(b.size() - fixed_state_size));
            fixed_A_ = A11 - A10 * A00.inverse() * A10.transpose();
            fixed_b_ = b1 - A10 * A00.inverse() * b0;
        } else {
            fixed_A_ = A;
            fixed_b_ = b;
        }

        // Remove fixed variables
        getStateVector();
        for (const auto& key : to_remove) {
            const auto& related_keys = related_var_keys_.at(key);
            for (const auto& related_key : related_keys) {
                related_var_keys_.at(related_key).erase(key);
            }
            related_var_keys_.erase(key);
            variables_.erase(key);
            if (variable_queue_.empty() || variable_queue_.front() != key) {
                throw std::runtime_error("[SlidingWindowFilter::marginalizeVariable] variable queue is not consistent");
            }
            variable_queue_.pop_front();
        }
        getStateVector();
    }

    // -----------------------------------------------------------------------------
    // addCostTerm
    // -----------------------------------------------------------------------------

    void SlidingWindowFilter::addCostTerm(const BaseCostTerm::ConstPtr &cost_term) {
        cost_terms_.emplace_back(cost_term);

        KeySet related_keys;
        cost_term->getRelatedVarKeys(related_keys);
        for (const auto &key : related_keys) {
            related_var_keys_.at(key).insert(related_keys.begin(), related_keys.end());
        }
    }

    // -----------------------------------------------------------------------------
    // cost
    // -----------------------------------------------------------------------------

    double SlidingWindowFilter::cost() const {

        // Sequential processing for small cost_terms_ to avoid parallel overhead
        if (cost_terms_.size() < 100) { // Tune threshold via profiling
            double cost = 0;
            for (size_t i = 0; i < cost_terms_.size(); i++) {
                try {
                    double cost_i = cost_terms_.at(i)->cost();
                    if (std::isnan(cost_i)) {
                        std::cerr << "[SlidingWindowFilter::cost] NaN cost term is ignored! " << std::endl;
                    } else {
                        cost += cost_i;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[SlidingWindowFilter::cost] exception in cost term: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "[SlidingWindowFilter::cost] exception in cost term: (unknown)" << std::endl;
                }
            }
            return cost;
        }

        // Parallel processing with TBB parallel_reduce for large cost_terms_
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads_);
        double total_cost = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, cost_terms_.size(), 100), 0.0, 
            [this](const tbb::blocked_range<size_t>& range, double init) -> double {
                double local_cost = init;
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    try {
                        double cost_i = cost_terms_.at(i)->cost();
                        if (std::isnan(cost_i)) {
                            std::cerr << "[SlidingWindowFilter::cost] NaN cost term is ignored! " << std::endl;
                        } else {
                            local_cost += cost_i;
                        }
                    } catch (const std::exception& e) {
            
                        std::cerr << "[SlidingWindowFilter::cost] exception in cost term at index " << i << ": " << e.what() << std::endl;
                    } catch (...) {
                        
                        std::cerr << "[SlidingWindowFilter::cost] exception in cost term at index " << i << ": (unknown)" << std::endl;
                    }
                }
                return local_cost; // Returns partial sum to TBB, not exiting cost
            },
            std::plus<double>() // Combine partial results
        );
        return total_cost;
    }

    // -----------------------------------------------------------------------------
    // getNumberOfCostTerms
    // -----------------------------------------------------------------------------

    unsigned int SlidingWindowFilter::getNumberOfCostTerms() const {
        return cost_terms_.size();
    }

    // -----------------------------------------------------------------------------
    // getNumberOfVariables
    // -----------------------------------------------------------------------------

    unsigned int SlidingWindowFilter::getNumberOfVariables() const {
        return variable_queue_.size();
    }

    // -----------------------------------------------------------------------------
    // getStateVector
    // -----------------------------------------------------------------------------

    StateVector::Ptr SlidingWindowFilter::getStateVector() const {
        *marginalize_state_vector_ = StateVector();
        *active_state_vector_ = StateVector();
        *state_vector_ = StateVector();

        // Iterate over variable_queue_ to maintain order
        bool marginalize = true;
        for (const auto &key : variable_queue_) {
            const auto &var = variables_.at(key);
            if (var.marginalize) {
                if (!marginalize) {
                    throw std::runtime_error("[SlidingWindowFilter::getStateVector] marginalized variables must be at the first");
                }
                marginalize_state_vector_->addStateVariable(var.variable);
            } else {
                marginalize = false;
                active_state_vector_->addStateVariable(var.variable);
            }
            state_vector_->addStateVariable(var.variable);
        }

        return active_state_vector_;
    }

    // -----------------------------------------------------------------------------
    // buildGaussNewtonTermss
    // -----------------------------------------------------------------------------

    void SlidingWindowFilter::buildGaussNewtonTerms(
        Eigen::SparseMatrix<double>& approximate_hessian,
        Eigen::VectorXd& gradient_vector) const {
        // Initialize block matrices
        std::vector<unsigned int> sqSizes = state_vector_->getStateBlockSizes();
        BlockSparseMatrix A_(sqSizes, true);
        BlockVector b_(sqSizes);

        // Process cost terms: sequential for small sizes, parallel for large
        if (cost_terms_.size() < 100) { // Tune threshold via profiling
            for (unsigned int c = 0; c < cost_terms_.size(); c++) {
                cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
            }
        } else {
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads_);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, cost_terms_.size(), 100),
                [&A_, &b_, this](const tbb::blocked_range<size_t>& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
                    }
                }
            );
        }

        // Convert to sparse Eigen types
        Eigen::MatrixXd Aupper(A_.toEigen(false));
        Eigen::MatrixXd A(Aupper.selfadjointView<Eigen::Upper>());
        Eigen::VectorXd b(b_.toEigen());

        // Add cached terms
        if (fixed_A_.size() > 0) {
            A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
            b.head(fixed_b_.size()) += fixed_b_;
        }

        // Marginalize fixed variables (Schur complement)
        const auto marginalize_state_size = marginalize_state_vector_->getStateSize();
        if (marginalize_state_size > 0) {
            // clang-format off
            Eigen::MatrixXd A00(A.topLeftCorner(marginalize_state_size, marginalize_state_size));
            Eigen::MatrixXd A10(A.bottomLeftCorner(A.rows() - marginalize_state_size, marginalize_state_size));
            Eigen::MatrixXd A11(A.bottomRightCorner(A.rows() - marginalize_state_size, A.cols() - marginalize_state_size));
            Eigen::VectorXd b0(b.head(marginalize_state_size));
            Eigen::VectorXd b1(b.tail(b.size() - marginalize_state_size));
            approximate_hessian = Eigen::MatrixXd(A11 - A10 * A00.llt().solve(A10.transpose())).sparseView();
            gradient_vector = b1 - A10 * A00.inverse() * b0;
            // clang-format on
        } else {
            approximate_hessian = A.sparseView();
            gradient_vector = b;
        }
    }
} // namespace finalicp



    