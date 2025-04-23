#pragma once

#include <Eigen/Core>

#include <problem/costterm/weightleastsqcostterm.hpp>
#include <solver/covariance.hpp>
#include <problem/problem.hpp>
#include <trajectory/constvel/variable.hpp>
#include <trajectory/interface.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace const_vel {

        class Interface : public traj::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;

                using PoseType = math::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;
                using CovType = Eigen::Matrix<double, 12, 12>;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                //Constructs an `AccelerationExtrapolator` instance.
                Interface(const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                //Checks if the extrapolator depends on any active variables
                void add(const Time time, const Evaluable<PoseType>::Ptr& T_k0, const Evaluable<VelocityType>::Ptr& w_0k_ink);

                //Retrieves related variable keys for factor graph optimization
                Variable::ConstPtr get(const Time time) const;

                //Retrieves interpolators for pose, velocity, and acceleration
                Evaluable<PoseType>::ConstPtr getPoseInterpolator(const Time time) const;
                Evaluable<VelocityType>::ConstPtr getVelocityInterpolator(const Time time) const;
                CovType getCovariance(const Covariance& cov, const Time time);

                //Adds prior constraints for pose, velocity, acceleration, and full state.
                void addPosePrior(const Time time, const PoseType& T_k0, const Eigen::Matrix<double, 6, 6>& cov);
                void addVelocityPrior(const Time time, const VelocityType& w_0k_ink, const Eigen::Matrix<double, 6, 6>& cov);
                void addStatePrior(const Time time, const PoseType& T_k0, const VelocityType& w_0k_ink, const CovType& cov);

                void addPriorCostTerms(Problem& problem) const;

            private:
            
                Eigen::Matrix<double, 6, 1> Qc_diag_;
                std::map<Time, Variable::Ptr> knot_map_;
                WeightedLeastSqCostTerm<6>::Ptr pose_prior_factor_ = nullptr;
                WeightedLeastSqCostTerm<6>::Ptr vel_prior_factor_ = nullptr;
                WeightedLeastSqCostTerm<12>::Ptr state_prior_factor_ = nullptr;
            };
        }  // namespace const_vel
    }  // namespace traj
}  // namespace finalicp