#pragma once

#include <iostream>

#include <problem/noisemodel/basenoisemodel.hpp>

namespace finalicp {

    template <int DIM>
    class StaticNoiseModel : public BaseNoiseModel<DIM> {
        public:
            using Ptr = std::shared_ptr<StaticNoiseModel<DIM>>;
            using ConstPtr = std::shared_ptr<const StaticNoiseModel<DIM>>;

            using MatrixT = typename BaseNoiseModel<DIM>::MatrixT;
            using VectorT = typename BaseNoiseModel<DIM>::VectorT;

            //Virtual destructor
            static Ptr MakeShared(const MatrixT& matrix, const NoiseType& type = NoiseType::COVARIANCE);

            //Constructs a DynamicNoiseModel.
            StaticNoiseModel(const MatrixT& matrix, const NoiseType& type = NoiseType::COVARIANCE);

            //Get the square root information matrix
            void setByCovariance(const MatrixT& matrix);

            //Compute the whitened error norm
            void setByInformation(const MatrixT& matrix);

            //Compute the whitened error vector
            void setBySqrtInformation(const MatrixT& matrix);

            MatrixT getSqrtInformation() const override;

            double getWhitenedErrorNorm(const VectorT& rawError) const override;

            VectorT whitenError(const VectorT& rawError) const override;

        private:

            //Convert covariance matrix to square root information matrix
            void assertPositiveDefiniteMatrix(const MatrixT& matrix) const;

            //Convert information matrix to square root information matrix
            MatrixT sqrtInformation_;
        
    };

    template <int DIM>
    auto StaticNoiseModel<DIM>::MakeShared(const MatrixT& matrix,
                                        const NoiseType& type) -> Ptr {
        return std::make_shared<StaticNoiseModel<DIM>>(matrix, type);
    }

    template <int DIM>
    StaticNoiseModel<DIM>::StaticNoiseModel(const MatrixT& matrix,
                                            const NoiseType& type) {
        // Depending on the type of 'matrix', we set the internal storage
        switch (type) {
            case NoiseType::COVARIANCE:
            setByCovariance(matrix);
            break;
            case NoiseType::INFORMATION:
            setByInformation(matrix);
            break;
            case NoiseType::SQRT_INFORMATION:
            setBySqrtInformation(matrix);
            break;
        }
    }

    template <int DIM>
    void StaticNoiseModel<DIM>::setByCovariance(const MatrixT& matrix) {
    // Information is the inverse of covariance
        setByInformation(matrix.inverse());
    }

    template <int DIM>
    void StaticNoiseModel<DIM>::setByInformation(const MatrixT& matrix) {
        // Check that the matrix is positive definite
        assertPositiveDefiniteMatrix(matrix);
        // Perform an LLT decomposition
        Eigen::LLT<MatrixT> lltOfInformation(matrix);
        // Store upper triangular matrix (the square root information matrix)
        setBySqrtInformation(lltOfInformation.matrixL().transpose());
    }

    template <int DIM>
    void StaticNoiseModel<DIM>::setBySqrtInformation(const MatrixT& matrix) {
        // Set internal storage matrix
        sqrtInformation_ = matrix;  // todo: check this is upper triangular
    }

    template <int DIM>
    auto StaticNoiseModel<DIM>::getSqrtInformation() const -> MatrixT {
        return sqrtInformation_;
    }

    template <int DIM>
    double StaticNoiseModel<DIM>::getWhitenedErrorNorm(const VectorT& rawError) const {
        return (sqrtInformation_ * rawError).norm();
    }

    template <int DIM>
    auto StaticNoiseModel<DIM>::whitenError(const VectorT& rawError) const
        -> VectorT {
        return sqrtInformation_ * rawError;
    }

    template <int DIM>
    void StaticNoiseModel<DIM>::assertPositiveDefiniteMatrix(
        const MatrixT& matrix) const {
        // Initialize an eigen value solver
        Eigen::SelfAdjointEigenSolver<MatrixT> eigsolver(matrix,
                                                        Eigen::EigenvaluesOnly);

        // Check the minimum eigen value
        if (eigsolver.eigenvalues().minCoeff() <= 0) {
            std::stringstream ss;
            ss << "Covariance \n"
            << matrix << "\n must be positive definite. "
            << "Min. eigenvalue : " << eigsolver.eigenvalues().minCoeff();
            throw std::invalid_argument(ss.str());
        }
    }
}  // namespace finalicp
