#pragma once

#include <algorithm>

#include <evaluable/evaluable.hpp>
#include <evaluable/jacobians.hpp>
#include <problem/costterm/basecostterm.hpp>
#include <problem/lossfunc/baselossfunc.hpp>
#include <problem/noisemodel/basenoisemodel.hpp>

#include <iostream>

namespace finalicp {
    template <int DIM>
    class WeightedLeastSqCostTerm : public BaseCostTerm {
        public:
            using Ptr = std::shared_ptr<WeightedLeastSqCostTerm<DIM>>;
            using ConstPtr = std::shared_ptr<const WeightedLeastSqCostTerm<DIM>>;

            using ErrorType = Eigen::Matrix<double, DIM, 1>;  // DIM is measurement dim

            static Ptr MakeShared(const typename Evaluable<ErrorType>::ConstPtr &error_function,
                                    const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
                                    const BaseLossFunc::ConstPtr &loss_function);

            WeightedLeastSqCostTerm(const typename Evaluable<ErrorType>::ConstPtr &error_function,
                                    const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
                                    const BaseLossFunc::ConstPtr &loss_function);

            double cost() const override;

            void getRelatedVarKeys(KeySet &keys) const override;

            void buildGaussNewtonTerms(const StateVector &state_vec,
                             BlockSparseMatrix *approximate_hessian,
                             BlockVector *gradient_vector) const override;
        private:
            ErrorType evalWeightedAndWhitened(Jacobians &jacobian_contaner) const;
            typename Evaluable<ErrorType>::ConstPtr error_function_;
            typename BaseNoiseModel<DIM>::ConstPtr noise_model_;
            BaseLossFunc::ConstPtr loss_function_;
    };

    template <int DIM>
    auto WeightedLeastSqCostTerm<DIM>::MakeShared(
            const typename Evaluable<ErrorType>::ConstPtr &error_function,
            const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
            const BaseLossFunc::ConstPtr &loss_function) -> Ptr {
        return std::make_shared<WeightedLeastSqCostTerm<DIM>>(
            error_function, noise_model, loss_function);
    }

    template <int DIM>
    WeightedLeastSqCostTerm<DIM>::WeightedLeastSqCostTerm(
        const typename Evaluable<ErrorType>::ConstPtr &error_function,
        const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
        const BaseLossFunc::ConstPtr &loss_function)
        : error_function_(error_function),
        noise_model_(noise_model),
        loss_function_(loss_function) {}

    template <int DIM>
    double WeightedLeastSqCostTerm<DIM>::cost() const {
        return loss_function_->cost(noise_model_->getWhitenedErrorNorm(error_function_->evaluate()));
    }

    template <int DIM>
    void WeightedLeastSqCostTerm<DIM>::getRelatedVarKeys(KeySet &keys) const {
        error_function_->getRelatedVarKeys(keys);
    }

    template <int DIM>
    void WeightedLeastSqCostTerm<DIM>::buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const {

        Jacobians jacobian_container;
        ErrorType error = this->evalWeightedAndWhitened(jacobian_container);
        const auto& jacobians = jacobian_container.get();

        // Get map keys into a vector for sorting
        std::vector<StateKey> keys;
        keys.reserve(jacobians.size());
        std::transform(jacobians.begin(), jacobians.end(), std::back_inserter(keys),[](const auto &pair) { return pair.first; });

        // Debug: Print number of keys and their values
        // ################################
        std::cout << "[DEBUG] Number of Jacobian keys: " << keys.size() << std::endl;
        for (const auto& key : keys) {
            std::cout << "[DEBUG] Processing key: " << key << std::endl;
        }
        // ################################

        for (size_t i = 0; i < keys.size(); ++i) {
            try {
                // Access Jacobian and block index
                const auto& key1 = keys.at(i);
                const auto& jac1 = jacobians.at(key1); // at for map safety
                unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

                // Debug: Print key, Jacobian size, and block index
                // ################################
                std::cout << "[DEBUG] i=" << i << ", key1=" << key1 << ", Jacobian rows=" << jac1.rows() << ", cols=" << jac1.cols() << ", blkIdx1=" << blkIdx1 << std::endl;
                // ################################

                // Compute gradient contribution
                Eigen::MatrixXd newGradTerm = (-1) * jac1.transpose() * error;

                // Debug: Print gradient term size and norm
                // ################################
                 std::cout << "[DEBUG] Gradient term size: (" << newGradTerm.rows() << ", " << newGradTerm.cols() << "), norm: " << newGradTerm.norm() << std::endl;
                // ################################

                // Update gradient 
                gradient_vector->mapAt(blkIdx1) += newGradTerm;

                // Inner loop for Hessian (upper triangle)
                for (size_t j = i; j < keys.size(); ++j) {
                    const auto& key2 = keys.at(j);
                    const auto& jac2 = jacobians.at(key2);
                    unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

                    // Debug: Print inner loop key and block index
                    // ################################
                    std::cout << "[DEBUG] j=" << j << ", key2=" << key2 << ", blkIdx2=" << blkIdx2 << std::endl;
                    // ################################

                    // Compute Hessian contribution
                    unsigned int row, col;
                    const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
                        if (blkIdx1 <= blkIdx2) {
                            row = blkIdx1;
                            col = blkIdx2;
                            return jac1.transpose() * jac2;
                        } else {
                            row = blkIdx2;
                            col = blkIdx1;
                            return jac2.transpose() * jac1;
                        }
                    }();

                    // Debug: Print Hessian term size and norm
                    // ################################
                    std::cout << "[DEBUG] Hessian term (row=" << row << ", col=" << col << ") size: (" << newHessianTerm.rows() << ", " << newHessianTerm.cols() << "), norm: " << newHessianTerm.norm() << std::endl;
                    // ################################

                    BlockSparseMatrix::BlockRowEntry& entry = approximate_hessian->rowEntryAt(row, col, true);
                    // omp_set_lock(&entry.lock);
                    entry.data += newHessianTerm;
                    // omp_unset_lock(&entry.lock);
                }
            } catch (const std::exception& e) {
                std::cerr << "[WeightedLeastSqCostTerm::buildGaussNewtonTerms] exception at index " << i << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[WeightedLeastSqCostTerm::buildGaussNewtonTerms] exception at index " << i << ": (unknown)" << std::endl;
            }
        }
    }

    template <int DIM>
    auto WeightedLeastSqCostTerm<DIM>::evalWeightedAndWhitened(Jacobians &jacobian_container) const -> ErrorType {
        // Initializes jacobian array
        jacobian_container.clear();

        // Debug: Print Hessian term size and norm
        // ################################
        std::cout << "[DEBUG] Cleared Jacobian container" << std::endl;
        // ################################

        // Get raw error and Jacobians
        ErrorType raw_error = error_function_->evaluate(noise_model_->getSqrtInformation(), jacobian_container);

        // Debug: Print Hessian term size and norm
        // ################################
        std::cout << "[DEBUG] Raw error norm: " << raw_error.norm() << ", size: (" << raw_error.rows() << ", " << raw_error.cols() << ")" << std::endl;
        std::cout << "[DEBUG] Number of Jacobians after evaluation: " << jacobian_container.get().size() << std::endl;
        // ################################

        // Get whitened error vector
        ErrorType white_error = noise_model_->whitenError(raw_error);

        // Debug: Print Hessian term size and norm
        // ################################
        std::cout << "[DEBUG] Whitened error norm: " << white_error.norm() << ", size: (" << white_error.rows() << ", " << white_error.cols() << ")" << std::endl;
        // ################################

        // Get weight from loss function
        double sqrt_w = sqrt(loss_function_->weight(white_error.norm()));

        // Debug: Print Hessian term size and norm
        // ################################
        std::cout << "[DEBUG] Sqrt weight (sqrt_w): " << sqrt_w << std::endl;
        // ################################

        // Weight the white Jacobians
        auto &jacobians = jacobian_container.get();
        for (auto &entry : jacobians) {

            // Debug: Print Hessian term size and norm
            // ################################
            std::cout << "[DEBUG] Weighting Jacobian for key: " << entry.first << ", original norm: " << entry.second.norm();
            // ################################

            entry.second *= sqrt_w;

            // Debug: Print Hessian term size and norm
            // ################################
            std::cout << ", new norm: " << entry.second.norm() << std::endl;
            // ################################
        }

        // Weight the error and return
        ErrorType weighted_error = sqrt_w * white_error;

        // Debug: Print Hessian term size and norm
        // ################################
        std::cout << "[DEBUG] Weighted error norm: " << weighted_error.norm() << ", size: (" << weighted_error.rows() << ", " << weighted_error.cols() << ")" << std::endl;
        // ################################

        return weighted_error;
    }
}  // namespace finalicp