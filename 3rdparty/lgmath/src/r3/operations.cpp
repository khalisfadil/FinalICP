#include "r3/operations.hpp"


namespace math {
    namespace r3 {
        
        // -----------------------------------------------------------------------------
        /// **Explicit instantiation for transformCovariance using se3::Transformation**.
        /// This version assumes **perfect certainty** in the transformation.
        template CovarianceMatrix transformCovariance<false>(
            const se3::Transformation& T_ba, 
            const CovarianceMatrixConstRef& cov_a, 
            const HPointConstRef& p_b);

        // -----------------------------------------------------------------------------
        /// **Explicit instantiations for transformCovariance using se3::TransformationWithCovariance**.
        /// These versions account for uncertainty in the transformation.

        /// - **THROW_IF_UNSET = true** → Ensures covariance is properly set before using it.
        template CovarianceMatrix transformCovariance<true>(
            const se3::TransformationWithCovariance& T_ba, 
            const CovarianceMatrixConstRef& cov_a, 
            const HPointConstRef& p_b);

        // -----------------------------------------------------------------------------
        /// - **THROW_IF_UNSET = false** → Allows transformations **without** covariance checks.
        template CovarianceMatrix transformCovariance<false>(
            const se3::TransformationWithCovariance& T_ba, 
            const CovarianceMatrixConstRef& cov_a, 
            const HPointConstRef& p_b);

    }  // namespace r3
} // namespace math
