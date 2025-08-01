#include <problem/optimizationproblem.hpp>

#include <iomanip>
#include <iostream>

namespace finalicp {

    // ##################################################
    // MakeShared
    // ##################################################

    auto OptimizationProblem::MakeShared(unsigned int num_threads)-> OptimizationProblem::Ptr {
        return std::make_shared<OptimizationProblem>(num_threads);
    }

    // ##################################################
    // OptimizationProblem
    // ##################################################

    OptimizationProblem::OptimizationProblem(unsigned int num_threads)
    : num_threads_(num_threads) {}

    // ##################################################
    // addStateVariable
    // ##################################################

    void OptimizationProblem::addStateVariable(const StateVarBase::Ptr &state) {
#ifdef DEBUG
        std::cout << "[OptimizationProblem DEBUG | addStateVariable] Adding State Variable. Key: " << state->key() << std::endl;
#endif
        state_vars_.push_back(state);
    }

    // ##################################################
    // addCostTerm
    // ##################################################

    void OptimizationProblem::addCostTerm(const BaseCostTerm::ConstPtr &costTerm) {
#ifdef DEBUG
        std::cout << "[OptimizationProblem DEBUG | addCostTerm] Adding Cost Term. Total now: " << cost_terms_.size() + 1 << std::endl;
#endif
        cost_terms_.push_back(costTerm);
    }

    // ##################################################
    // getNumberOfCostTerms
    // ##################################################

    unsigned int OptimizationProblem::getNumberOfCostTerms() const {
        return cost_terms_.size();
    }

    // ##################################################
    // cost
    // ##################################################

    double OptimizationProblem::cost() const {

        // Sequential processing for small cost_terms_ to avoid parallel overhead
#ifdef DEBUG
        std::cout << "[OptimizationProblem DEBUG | cost] Calculating total cost from " << cost_terms_.size() << " cost terms..." << std::endl;
#endif
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
    }

    // ##################################################
    // getStateVector
    // ##################################################

    StateVector::Ptr OptimizationProblem::getStateVector() const {
        *state_vector_ = StateVector();

#ifdef DEBUG
        std::cout << "[OptimizationProblem DEBUG | getStateVector] assembling StateVector from " << state_vars_.size() << " total variables..." << std::endl;
        int unlocked_count = 0;
#endif

        for (const auto &state_var : state_vars_) {
            if (!state_var->locked()) {
                state_vector_->addStateVariable(state_var);
#ifdef DEBUG
                unlocked_count++;
#endif
            }
#ifdef DEBUG
            else {
                std::cout << "    - Skipping locked variable with key: " << state_var->key() << std::endl;
            }
#endif
        }
#ifdef DEBUG
        std::cout << "    - Assembly complete. Added " << unlocked_count << " unlocked variables to the active state." << std::endl;
#endif
        return state_vector_;
    }

    // ##################################################
    // buildGaussNewtonTerms
    // ##################################################

    void OptimizationProblem::buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian, Eigen::VectorXd& gradient_vector) const {
#ifdef DEBUG
        std::cout << "[OptimizationProblem DEBUG | buildGaussNewtonTerms] Building Gauss-Newton system from " << cost_terms_.size() << " cost terms." << std::endl;
#endif
        std::vector<unsigned int> sqSizes = state_vector_->getStateBlockSizes();
        
        BlockSparseMatrix A_(sqSizes, true);
        BlockVector b_(sqSizes);

        // Process cost terms: sequential for small sizes, parallel for large
        
        for (unsigned int c = 0; c < cost_terms_.size(); c++) {
            try {

                cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
            } catch (const std::exception& e) {
                std::cerr << "[OptimizationProblem::buildGaussNewtonTerms] exception in cost term: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[OptimizationProblem::buildGaussNewtonTerms]  exception in cost term: (unknown)" << std::endl;
            }
        }

        approximate_hessian = A_.toEigen(false);
        gradient_vector = b_.toEigen();

#ifdef DEBUG
        // --- [IMPROVEMENT] Sanity-check the final constructed system ---
        bool hessian_ok = approximate_hessian.coeffs().allFinite();
        bool gradient_ok = gradient_vector.allFinite();

        if (!hessian_ok) {
            std::cerr << "[OptimizationProblem DEBUG | buildGaussNewtonTerms] CRITICAL: Assembled Hessian contains non-finite values!" << std::endl;
        }
        if (!gradient_ok) {
            std::cerr << "[OptimizationProblem DEBUG | buildGaussNewtonTerms] CRITICAL: Assembled Gradient contains non-finite values!" << std::endl;
        }
        if (hessian_ok && gradient_ok) {
            std::cout << "    - System built successfully. Hessian non-zeros: " << approximate_hessian.nonZeros()
                      << ", Gradient norm: " << gradient_vector.norm() << std::endl;
        }
#endif
    }
} // namespace finalicp