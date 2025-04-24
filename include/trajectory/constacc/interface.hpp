#pragma once

#include <Eigen/Core>

#include <problem/costterm/weightleastsqcostterm.hpp>
#include <problem/problem.hpp>
#include <solver/covariance.hpp>
#include <trajectory/constacc/variable.hpp>
#include <trajectory/interface.hpp>
#include <trajectory/time.hpp>

namespace finalicp {
    namespace traj {
        namespace const_acc {

        class Interface : public traj::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;

                using PoseType = math::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;
                using AccelerationType = Eigen::Matrix<double, 6, 1>;
                using CovType = Eigen::Matrix<double, 18, 18>;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                //Constructs an `AccelerationExtrapolator` instance.
                Interface(const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones());

                //Checks if the extrapolator depends on any active variables
                void add(const Time time, const Evaluable<PoseType>::Ptr& T_k0, const Evaluable<VelocityType>::Ptr& w_0k_ink, const Evaluable<AccelerationType>::Ptr& dw_0k_ink);

                //Retrieves related variable keys for factor graph optimization
                Variable::ConstPtr get(const Time time) const;

                //Retrieves interpolators for pose, velocity, and acceleration
                Evaluable<PoseType>::ConstPtr getPoseInterpolator(const Time time) const;
                Evaluable<VelocityType>::ConstPtr getVelocityInterpolator(const Time time) const;
                Evaluable<AccelerationType>::ConstPtr getAccelerationInterpolator(const Time time) const;
                CovType getCovariance(const Covariance& cov, const Time time);

                //Adds prior constraints for pose, velocity, acceleration, and full state.
                void addPosePrior(const Time time, const PoseType& T_k0, const Eigen::Matrix<double, 6, 6>& cov);
                void addVelocityPrior(const Time time, const VelocityType& w_0k_ink, const Eigen::Matrix<double, 6, 6>& cov);
                void addAccelerationPrior(const Time time, const AccelerationType& dw_0k_ink, const Eigen::Matrix<double, 6, 6>& cov);
                void addStatePrior(const Time time, const PoseType& T_k0,const VelocityType& w_0k_ink, const AccelerationType& dw_0k_ink, const CovType& cov);

                void addPriorCostTerms(Problem& problem) const;

                Eigen::Matrix<double, 6, 1> Qc_diag_;
                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt) const;
                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt) const;
                Eigen::Matrix<double, 18, 18> getTranPublic(const double& dt) const;

            protected:
            
                std::map<Time, Variable::Ptr> knot_map_;                            //map storing trajectory knots

                //Weighted least-squares cost terms for pose, velocity, acceleration, and full state
                WeightedLeastSqCostTerm<6>::Ptr pose_prior_factor_ = nullptr;
                WeightedLeastSqCostTerm<6>::Ptr vel_prior_factor_ = nullptr;
                WeightedLeastSqCostTerm<6>::Ptr acc_prior_factor_ = nullptr;
                WeightedLeastSqCostTerm<18>::Ptr state_prior_factor_ = nullptr;

                //Internal methods for Jacobian calculations.
                Eigen::Matrix<double, 18, 18> getJacKnot1_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
                Eigen::Matrix<double, 18, 18> getJacKnot2_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;

                //Internal process noise covariance computations.
                Eigen::Matrix<double, 18, 18> getQ_(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;
                Eigen::Matrix<double, 18, 18> getQinv_(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const;

                //Internal methods for interpolators.
                Evaluable<PoseType>::Ptr getPoseInterpolator_(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
                Evaluable<VelocityType>::Ptr getVelocityInterpolator_(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
                Evaluable<AccelerationType>::Ptr getAccelerationInterpolator_(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;

                //Internal methods for extrapolators.
                Evaluable<PoseType>::Ptr getPoseExtrapolator_(const Time time, const Variable::ConstPtr& knot) const;
                Evaluable<VelocityType>::Ptr getVelocityExtrapolator_(const Time time, const Variable::ConstPtr& knot) const;
                Evaluable<AccelerationType>::Ptr getAccelerationExtrapolator_(const Time time, const Variable::ConstPtr& knot) const;

                //Internal method to compute prior factor.
                Evaluable<Eigen::Matrix<double, 18, 1>>::Ptr getPriorFactor_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const;
            };
        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp