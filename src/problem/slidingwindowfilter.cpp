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
// #ifdef DEBUG
//             std::cout << "[SlidingWindowFilter DEBUG | addStateVariable] Adding State Variable. Key: " << var->key() << std::endl;
// #endif
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

    void SlidingWindowFilter::marginalizeVariable(const std::vector<StateVarBase::Ptr> &variables) {
        if (variables.empty()) return;

#ifdef DEBUG
            std::cout << "\n[000# SlidingWindowFilter DEBUG | marginalizeVariable]  ###################### START ####################. \n" << std::endl;
            std::cout << "[001# SlidingWindowFilter DEBUG | marginalizeVariable] Starting Marginalization for " << variables.size() << " variables" << std::endl;
#endif

        ///
        for (const auto &variable : variables) {
            variables_.at(variable->key()).marginalize = true;
        }

        /// remove fixed variables from the queue

        StateVector fixed_state_vector;
        StateVector state_vector;

        //
        std::vector<StateKey> to_remove;
        bool fixed = true;
        for (const auto &key : variable_queue_) {
            const auto &var = variables_.at(key);
            const auto &related_keys = related_var_keys_.at(key);
            // If all of a variable's related keys are also variables to be
            // marginalized, then add the key to to_remove
            if (std::all_of(related_keys.begin(), related_keys.end(),
                            [this](const StateKey &key) {
                            return variables_.at(key).marginalize;
                            })) {
            if (!fixed) {
                throw std::runtime_error("fixed variables must be at the first");
            }
            fixed_state_vector.addStateVariable(var.variable);
            to_remove.emplace_back(key);
            } else {
            fixed = false;
            }
            state_vector.addStateVariable(var.variable);
        }

#ifdef DEBUG
        std::cout << "[002# SlidingWindowFilter DEBUG | marginalizeVariable] Identified " << to_remove.size() << " variables to be marginalized out." << std::endl;
        std::cout << "[003# SlidingWindowFilter DEBUG | marginalizeVariable] Total variables in current problem: " << state_vector.getNumberOfStates() << std::endl;
#endif

        //
        std::vector<BaseCostTerm::ConstPtr> active_cost_terms;
        active_cost_terms.reserve(cost_terms_.size());
        //
        const auto state_sizes = state_vector.getStateBlockSizes();
        BlockSparseMatrix A_(state_sizes, true);
        BlockVector b_(state_sizes);

        for (unsigned int c = 0; c < cost_terms_.size(); c++) {
            KeySet keys;
            cost_terms_.at(c)->getRelatedVarKeys(keys);
            // build A-b using only the cost terms where all the variables
            // involved are to be marginalized.
            if (std::all_of(keys.begin(), keys.end(), [this](const StateKey &key) {
                return variables_.at(key).marginalize;
                })) {
            cost_terms_.at(c)->buildGaussNewtonTerms(state_vector, &A_, &b_);
            } else {
            // #pragma omp critical(active_cost_terms_update)
            { active_cost_terms.emplace_back(cost_terms_.at(c)); }
            }
        }
        //
        cost_terms_ = active_cost_terms;

#ifdef DEBUG
        std::cout << "[004# SlidingWindowFilter DEBUG | marginalizeVariable] Built system with " << active_cost_terms.size() << " remaining active cost terms." << std::endl;
#endif
        /// \todo use sparse matrix
        Eigen::MatrixXd Aupper(A_.toEigen(false));
        Eigen::MatrixXd A(Aupper.selfadjointView<Eigen::Upper>());
        Eigen::VectorXd b(b_.toEigen());

        // add the cached terms (always top-left block)
        if (fixed_A_.size() > 0) {
#ifdef DEBUG
            std::cout << "[005# SlidingWindowFilter DEBUG | marginalizeVariable] Applying prior from previous marginalization (Size: " << fixed_A_.rows() << "x" << fixed_A_.cols() << ")" << std::endl;
#endif
            A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
            b.head(fixed_b_.size()) += fixed_b_;
        }

        // marginalize the fixed variables
        const auto fixed_state_size = fixed_state_vector.getStateSize();
        if (fixed_state_size > 0) {
#ifdef DEBUG
            std::cout << "[006# SlidingWindowFilter DEBUG | marginalizeVariable] Performing Schur complement. Marginalizing " << fixed_state_size << " dimensions." << std::endl;
#endif
            Eigen::MatrixXd A00(A.topLeftCorner(fixed_state_size, fixed_state_size));
#ifdef DEBUG
            Eigen::FullPivLU<Eigen::MatrixXd> lu(A00);
            if (!lu.isInvertible()) {
                std::cerr << "[007# SlidingWindowFilter DEBUG | marginalizeVariable] CRITICAL: Matrix block A00 is not invertible during marginalization! System may be ill-conditioned." << std::endl;
                // You might want to throw an exception here in a real application
            }
#endif
            Eigen::MatrixXd A10(A.bottomLeftCorner(A.rows() - fixed_state_size, fixed_state_size));
            Eigen::MatrixXd A11(A.bottomRightCorner(A.rows() - fixed_state_size, A.cols() - fixed_state_size));
            Eigen::VectorXd b0(b.head(fixed_state_size));
            Eigen::VectorXd b1(b.tail(b.size() - fixed_state_size));
            fixed_A_ = A11 - A10 * A00.inverse() * A10.transpose();
            fixed_b_ = b1 - A10 * A00.inverse() * b0;
#ifdef DEBUG
            std::cout << "[008# SlidingWindowFilter DEBUG | marginalizeVariable] New prior created. Size: " << fixed_A_.rows() << "x" << fixed_A_.cols() << ", Norm: " << fixed_A_.norm() << std::endl;
#endif
        } else {
            fixed_A_ = A;
            fixed_b_ = b;
        }

        /// remove the fixed variables
        getStateVector();
        for (const auto &key : to_remove) {
            const auto related_keys = related_var_keys_.at(key);
            for (const auto &related_key : related_keys) {
            related_var_keys_.at(related_key).erase(key);
            }
            related_var_keys_.erase(key);
            variables_.erase(key);
            if (variable_queue_.empty() || variable_queue_.front() != key)
            throw std::runtime_error("variable queue is not consistent");
            variable_queue_.pop_front();
        }
        getStateVector();
#ifdef DEBUG
        std::cout << "[009# SlidingWindowFilter DEBUG | marginalizeVariable] Marginalization Complete. " << variables_.size() << " variables remain." << std::endl;
        std::cout << "\n[000# SlidingWindowFilter DEBUG | marginalizeVariable]  ###################### END ####################. \n" << std::endl;
#endif
    }

    // -----------------------------------------------------------------------------
    // addCostTerm
    // -----------------------------------------------------------------------------

    void SlidingWindowFilter::addCostTerm(const BaseCostTerm::ConstPtr &cost_term) {
        cost_terms_.emplace_back(cost_term);
        KeySet related_keys;
        cost_term->getRelatedVarKeys(related_keys);
// #ifdef DEBUG
//         // --- [IMPROVEMENT] Log cost term addition and its connections ---
//         std::stringstream ss;
//         ss << "[SlidingWindowFilter DEBUG | addCostTerm] Adding Cost Term. Connects keys: { ";
//         for(const auto& key : related_keys) { ss << key << " "; }
//         ss << "}";
//         std::cout << ss.str() << std::endl;
// #endif
        for (const auto &key : related_keys) {
            related_var_keys_.at(key).insert(related_keys.begin(), related_keys.end());
        }
    }

    // -----------------------------------------------------------------------------
    // cost
    // -----------------------------------------------------------------------------

    double SlidingWindowFilter::cost() const {

        // Sequential processing for small cost_terms_ to avoid parallel overhead
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
// #ifdef DEBUG
//         // --- [IMPROVEMENT] Log the partitioning of the state vector ---
//         std::cout << "[SlidingWindowFilter DEBUG | getStateVector] getStateVector called. Partitioning: "
//                 << marginalize_state_vector_->getNumberOfStates() << " to be marginalized, "
//                 << active_state_vector_->getNumberOfStates() << " active." << std::endl;
// #endif
        return active_state_vector_;
    }

    // -----------------------------------------------------------------------------
    // buildGaussNewtonTermss
    // -----------------------------------------------------------------------------

    void SlidingWindowFilter::buildGaussNewtonTerms(
            Eigen::SparseMatrix<double> &approximate_hessian,
            Eigen::VectorXd &gradient_vector) const {
#ifdef DEBUG
        std::cout << "\n[000# SlidingWindowFilter DEBUG | buildGaussNewtonTerms]  ###################### START ####################. \n" << std::endl;
        std::cout << "[001# SlidingWindowFilter DEBUG | buildGaussNewtonTerms] buildGaussNewtonTerms called for the active window with cost_terms_ size: " << cost_terms_.size() << std::endl;
#endif
        //
        std::vector<unsigned int> sqSizes = state_vector_->getStateBlockSizes();
        BlockSparseMatrix A_(sqSizes, true);
        BlockVector b_(sqSizes);

        for (unsigned int c = 0; c < cost_terms_.size(); c++) {
            cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
        }

        // Convert to Eigen Types
        Eigen::MatrixXd Aupper(A_.toEigen(false));
        Eigen::MatrixXd A(Aupper.selfadjointView<Eigen::Upper>());
        Eigen::VectorXd b(b_.toEigen());

// #ifdef DEBUG
//         // Print the matrix and vector immediately after they are constructed from cost terms
//         std::cout << "[SlidingWindowFilter DEBUG | buildGaussNewtonTerms] Matrix A (Hessian) after accumulating cost terms ("
//                   << A.rows() << "x" << A.cols() << "):" << std::endl;
//         std::cout << A << std::endl;
//         std::cout << "[SlidingWindowFilter DEBUG | buildGaussNewtonTerms] Vector b (gradient) after accumulating cost terms ("
//                   << b.size() << "x1):" << std::endl;
//         std::cout << b.transpose() << std::endl; // Print transpose for better console layout
// #endif

        if (fixed_A_.size() > 0) {
#ifdef DEBUG
            std::cout << "[002# SlidingWindowFilter DEBUG | buildGaussNewtonTerms] Applying prior from marginalization (Size: " << fixed_A_.rows() << "x" << fixed_A_.cols() << ")" << std::endl;
#endif
            A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
            b.head(fixed_b_.size()) += fixed_b_;

#ifdef DEBUG
            // Print the matrix and vector again after applying the marginalization prior
            // std::cout << "[SlidingWindowFilter DEBUG | buildGaussNewtonTerms] Matrix A after applying marginalization prior:" << std::endl;
            // std::cout << A << std::endl;
            // std::cout << "[SlidingWindowFilter DEBUG | buildGaussNewtonTerms] Vector b after applying marginalization prior:" << std::endl;
            // std::cout << b.transpose() << std::endl;
#endif
        }

        // marginalize the fixed variables
        const auto marginalize_state_size = marginalize_state_vector_->getStateSize();
        if (marginalize_state_size > 0) {
#ifdef DEBUG
            std::cout << "[003# SlidingWindowFilter DEBUG | buildGaussNewtonTerms] buildGaussNewtonTerms is marginalizing " << marginalize_state_size << " dimensions to form final system." << std::endl;
#endif
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
#ifdef DEBUG
        std::cout << "[005# SlidingWindowFilter DEBUG | buildGaussNewtonTerms] Final system built. Hessian non-zeros: " << approximate_hessian.nonZeros()
                << ", Gradient norm: " << gradient_vector.norm() << std::endl;
        std::cout << "\n[000# SlidingWindowFilter DEBUG | buildGaussNewtonTerms]  ###################### END ####################. \n" << std::endl;
#endif
    }
} // namespace finalicp



    