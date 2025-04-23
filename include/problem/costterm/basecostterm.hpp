#pragma once

#include <matrixoperator/matrixsparse.hpp>
#include <matrixoperator/vector.hpp> 

#include <problem/statevector.hpp>

namespace finalicp {
    //Interface for a cost function term contributing to the objective function.
    class BaseCostTerm {
        public:
            using Ptr = std::shared_ptr<BaseCostTerm>;
            using ConstPtr = std::shared_ptr<const BaseCostTerm>;

            //Explicit virtual destructor for safe inheritance
            virtual ~BaseCostTerm() = default;

            //Computes the cost contribution to the objective function.
            virtual double cost() const = 0;

            using KeySet = std::unordered_set<StateKey, StateKeyHash>;
            //Retrieves variable keys that this cost term depends on.
            virtual void getRelatedVarKeys(KeySet &keys) const = 0;

            //Adds contributions to the Gauss-Newton system.
            virtual void buildGaussNewtonTerms(const StateVector &state_vec,
                                     BlockSparseMatrix *approximate_hessian,
                                     BlockVector *gradient_vector) const = 0;
    };
} // namespace finalicp