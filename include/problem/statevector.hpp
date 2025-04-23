#pragma once

#include <map>
#include <vector>

#include <evaluable/statekey.hpp> 
#include <evaluable/statevar.hpp>

namespace finalicp {
    //Container for managing state variables in optimization.
    class StateVector {
        public:
            using Ptr = std::shared_ptr<StateVector>;
            using ConstPtr = std::shared_ptr<const StateVector>;
            using WeakPtr = std::weak_ptr<StateVector>;
            using ConstWeakPtr = std::weak_ptr<const StateVector>;

            static Ptr MakeShared() { return std::make_shared<StateVector>(); }

            //Performs a deep copy of the state vector
            StateVector clone() const;

            //Copies values from another state vector.
            void copyValues(const StateVector &other);

            //Adds a state variable to the vector.
            void addStateVariable(const StateVarBase::Ptr &statevar);

            //Checks if a state variable exists.
            bool hasStateVariable(const StateKey &key) const;

            //Retrieves a state variable by key.
            StateVarBase::ConstPtr getStateVariable(const StateKey &key) const;

            //Returns the total number of state variables
            unsigned int getNumberOfStates() const;

            //Returns the block index of a specific state.
            int getStateBlockIndex(const StateKey &key) const;

            //Returns an ordered list of block sizes.
            std::vector<unsigned int> getStateBlockSizes() const;
            
            //Returns the total size of the state vector.
            unsigned int getStateSize() const;

            //Updates the state vector using a perturbation.
            void update(const Eigen::VectorXd &perturbation);

        private:

            //Container for state variables and indexing.
            struct StateContainer {
                /// State
                StateVarBase::Ptr state;

                /// Block index in active state (set to -1 if not an active variable)
                int local_block_index;
            };

            //Thread-safe storage for state variables.
            using StateMap = std::unordered_map<StateKey, StateContainer, StateKeyHash>;
            StateMap states_;

            //Total number of block entries in the state vector (atomic for thread safety).
            unsigned int num_block_entries_ = 0;

    };
} // namespace finalicp