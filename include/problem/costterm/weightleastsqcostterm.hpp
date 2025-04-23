#pragma once

#include <algorithm>

#include <evaluable/evaluable.hpp>
#include <evaluable/jacobians.hpp>
#include <problem/costterm/basecostterm.hpp>
#include <problem/lossfunc/baselossfunc.hpp>
#include <problem/noisemodel/basenoisemodel.hpp>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>

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
    void WeightedLeastSqCostTerm<DIM>::buildGaussNewtonTerms(
            const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
            BlockVector *gradient_vector) const {

        // Compute weighted and whitened errors and Jacobians
        Jacobians jacobian_container;
        ErrorType error = this->evalWeightedAndWhitened(jacobian_container);
        const auto& jacobians = jacobian_container.get();

        // Get map keys into a vector for sorting
        std::vector<StateKey> keys;
        keys.reserve(jacobians.size());
        std::transform(jacobians.begin(), jacobians.end(), std::back_inserter(keys),
                        [](const auto &pair) { return pair.first; });

        // Track exceptions for thread-safe logging
        std::atomic<size_t> exception_count{0};

        // Parallelize outer loop over Jacobians
        tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size(), 1),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    try {
                        // Access Jacobian and block index
                        const auto& key1 = keys.at(i);
                        const auto& jac1 = jacobians.at(key1); // at for map safety
                        unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

                        // Compute gradient contribution
                        Eigen::MatrixXd newGradTerm = (-1) * jac1.transpose() * error;

                        // Update gradient (thread-safe with mutex)
                        static tbb::spin_mutex grad_mutex;
                        tbb::spin_mutex::scoped_lock lock(grad_mutex);
                        gradient_vector->mapAt(blkIdx1) += newGradTerm;

                        // Inner loop for Hessian (upper triangle)
                        for (size_t j = i; j < keys.size(); ++j) {
                            const auto& key2 = keys.at(j);
                            const auto& jac2 = jacobians.at(key2);
                            unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

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

                            // Update Hessian (thread-safe with BlockRowEntry lock)
                            BlockSparseMatrix::BlockRowEntry& entry =
                                approximate_hessian->rowEntryAt(row, col, true);
                            tbb::spin_mutex::scoped_lock entry_lock(entry.lock);
                            entry.data += newHessianTerm;
                        }
                    } catch (const std::exception& e) {
                        ++exception_count;
                        std::cerr << "[WeightedLeastSqCostTerm::buildGaussNewtonTerms] exception at index " << i << ": " << e.what() << std::endl;
                    } catch (...) {
                        ++exception_count;
                        std::cerr << "[WeightedLeastSqCostTerm::buildGaussNewtonTerms] exception at index " << i << ": (unknown)" << std::endl;
                    }
                }
            }
        );
    }

    template <int DIM>
    auto WeightedLeastSqCostTerm<DIM>::evalWeightedAndWhitened(
        Jacobians &jacobian_contaner) const -> ErrorType {
        // initializes jacobian array
        jacobian_contaner.clear();

        // Get raw error and Jacobians
        ErrorType raw_error = error_function_->evaluate(
            noise_model_->getSqrtInformation(), jacobian_contaner);

        // Get whitened error vector
        ErrorType white_error = noise_model_->whitenError(raw_error);

        // Get weight from loss function
        double sqrt_w = sqrt(loss_function_->weight(white_error.norm()));

        // Weight the white jacobians
        auto &jacobians = jacobian_contaner.get();
        for (auto &entry : jacobians) entry.second *= sqrt_w;

        // Weight the error and return
        return sqrt_w * white_error;
    }

}  // namespace finalicp