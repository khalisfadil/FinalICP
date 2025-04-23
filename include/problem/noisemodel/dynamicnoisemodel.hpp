#pragma once

#include <iostream>

#include <problem/noisemodel/basenoisemodel.hpp>
#include <evaluable/evaluable.hpp>

namespace finalicp {

    template <int DIM>
    class DynamicNoiseModel : public BaseNoiseModel<DIM> {
        public:
            using Ptr = std::shared_ptr<DynamicNoiseModel<DIM>>;
            using ConstPtr = std::shared_ptr<const DynamicNoiseModel<DIM>>;

            using MatrixT = Eigen::Matrix<double, DIM, DIM>;
            using VectorT = Eigen::Matrix<double, DIM, 1>;
            using MatrixTEvalPtr = typename Evaluable<MatrixT>::ConstPtr;

            //Virtual destructor
            static Ptr MakeShared(const MatrixTEvalPtr eval, const NoiseType type = NoiseType::COVARIANCE);

            //Constructs a DynamicNoiseModel.
            DynamicNoiseModel(const MatrixTEvalPtr eval, const NoiseType type = NoiseType::COVARIANCE);

            //Get the square root information matrix
            MatrixT getSqrtInformation() const override;

            //Compute the whitened error norm
            double getWhitenedErrorNorm(const VectorT& rawError) const override;

            //Compute the whitened error vector
            VectorT whitenError(const VectorT& rawError) const override;

        private:

            //Convert covariance matrix to square root information matrix
            MatrixT setByCovariance(const MatrixT& matrix) const;

            //Convert information matrix to square root information matrix
            MatrixT setByInformation(const MatrixT& matrix) const;

            //Validate and return the square root information matrix
            MatrixT setBySqrtInformation(const MatrixT& matrix) const;

            //Validate if a matrix is positive definite
            void assertPositiveDefiniteMatrix(const MatrixT& matrix) const;

            //Evaluable providing the noise matrix
            const MatrixTEvalPtr eval_;

            //The noise representation type
            NoiseType type_;
        
    };

    template <int DIM>
    auto DynamicNoiseModel<DIM>::MakeShared(const MatrixTEvalPtr eval,
                                        const NoiseType type) -> Ptr {
        return std::make_shared<DynamicNoiseModel<DIM>>(eval, type);
    }

    template <int DIM>
    DynamicNoiseModel<DIM>::DynamicNoiseModel(const MatrixTEvalPtr eval,
                                            const NoiseType type) : eval_(eval) {
                                            type_ = type;
                                            if (eval_ == nullptr)
                                                std::cout << "Evaluator initialization failed somehow";
                                            }

    template <int DIM>
    auto DynamicNoiseModel<DIM>::setByCovariance(const MatrixT& matrix) const -> MatrixT {
    // Information is the inverse of covariance
        return setByInformation(matrix.inverse());
    }

    template <int DIM>
    auto DynamicNoiseModel<DIM>::setByInformation(const MatrixT& matrix) const -> MatrixT{
        // Check that the matrix is positive definite
        assertPositiveDefiniteMatrix(matrix);
        // Perform an LLT decomposition
        Eigen::LLT<MatrixT> lltOfInformation(matrix);
        // Store upper triangular matrix (the square root information matrix)
        return setBySqrtInformation(lltOfInformation.matrixL().transpose());
    }

    template <int DIM>
    auto DynamicNoiseModel<DIM>::setBySqrtInformation(const MatrixT& matrix) const -> MatrixT{
        return matrix;  // todo: check this is upper triangular
    }

    template <int DIM>
    auto DynamicNoiseModel<DIM>::getSqrtInformation() const -> MatrixT {
        const MatrixT matrix = eval_->value();
        switch (type_) {
            case NoiseType::INFORMATION:
                return setByInformation(matrix);
            case NoiseType::SQRT_INFORMATION:
                return setBySqrtInformation(matrix);
            case NoiseType::COVARIANCE:
            default:
                return setByCovariance(matrix);
        }
    }

    template <int DIM>
    double DynamicNoiseModel<DIM>::getWhitenedErrorNorm(
        const VectorT& rawError) const {
        return (getSqrtInformation() * rawError).norm();
    }

    template <int DIM>
    auto DynamicNoiseModel<DIM>::whitenError(const VectorT& rawError) const
        -> VectorT {
        return getSqrtInformation() * rawError;
    }

    template <int DIM>
    void DynamicNoiseModel<DIM>::assertPositiveDefiniteMatrix(const MatrixT& matrix) const {
        // Initialize an eigen value solver
        Eigen::SelfAdjointEigenSolver<MatrixT> eigsolver(matrix,
                                                        Eigen::EigenvaluesOnly);

        // Check the minimum eigen value
        if (eigsolver.eigenvalues().minCoeff() <= 0) {
            std::stringstream ss;
            ss << "Covariance \n"
            << matrix << "\n must be positive definite. "
            << "Min. eigenvalue : " << eigsolver.eigenvalues().minCoeff() << std::endl;
            throw std::invalid_argument(ss.str());
        }
    }

}  // namespace finalicp
