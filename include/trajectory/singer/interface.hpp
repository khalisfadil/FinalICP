#pragma once

#include <Eigen/Core>

#include <trajectory/constacc/interface.hpp>
#include <trajectory/time.hpp>

#include <trajectory/singer/accelerationextrapolator.hpp>
#include <trajectory/singer/accelerationinterpolator.hpp>
#include <trajectory/singer/helper.hpp>
#include <trajectory/singer/poseextrapolator.hpp>
#include <trajectory/singer/poseinterpolator.hpp>
#include <trajectory/singer/priorfactor.hpp>
#include <trajectory/singer/velocityextrapolator.hpp>
#include <trajectory/singer/velocityinterpolator.hpp>


namespace finalicp {
    namespace traj {
        namespace singer {

        class Interface : public traj::const_acc::Interface {
            public:
                using Ptr = std::shared_ptr<Interface>;
                using ConstPtr = std::shared_ptr<const Interface>;
                using Variable = traj::const_acc::Variable;

                using PoseType = math::se3::Transformation;
                using VelocityType = Eigen::Matrix<double, 6, 1>;
                using AccelerationType = Eigen::Matrix<double, 6, 1>;

                //Factory method to create a shared instance of Variable.
                static Ptr MakeShared(const Eigen::Matrix<double, 6, 1>& alpha_diag = Eigen::Matrix<double, 6, 1>::Ones(), const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones()) {
                    return std::make_shared<Interface>(alpha_diag, Qc_diag);
                }

                //Constructs an `AccelerationExtrapolator` instance.
                Interface(const Eigen::Matrix<double, 6, 1>& alpha_diag = Eigen::Matrix<double, 6, 1>::Ones(), const Eigen::Matrix<double, 6, 1>& Qc_diag = Eigen::Matrix<double, 6, 1>::Ones())
                    : traj::const_acc::Interface(Qc_diag), alpha_diag_(alpha_diag) {
#ifdef DEBUG
                    // --- [IMPROVEMENT] Log creation and configuration parameters ---
                    std::cout << "🎶 [SINGER DEBUG | Interface] Creating Singer::Interface." << std::endl;
                    std::cout << "    - Alpha Diag: " << alpha_diag_.transpose() << std::endl;
                    std::cout << "    - Qc Diag:    " << Qc_diag_.transpose() << std::endl;
#endif
                    }

                //Checks if the extrapolator depends on any active variables
                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                    return getQ(dt, alpha_diag_, Qc_diag).inverse();
                }

                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
                    return getQ(dt, alpha_diag_, Qc_diag);
                }

                Eigen::Matrix<double, 18, 18> getQinvPublic(const double& dt) const {
                    return getQ(dt, alpha_diag_, Qc_diag_).inverse();
                }

                Eigen::Matrix<double, 18, 18> getQPublic(const double& dt) const {
                    return getQ(dt, alpha_diag_, Qc_diag_);
                }

                Eigen::Matrix<double, 18, 18> getTranPublic(const double& dt) const {
                    return getTran(dt, alpha_diag_);
                }

            protected:

                Eigen::Matrix<double, 6, 1> alpha_diag_;

                Eigen::Matrix<double, 18, 18> getJacKnot1_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                    return getJacKnot1(knot1, knot2, alpha_diag_);
                }

                Eigen::Matrix<double, 18, 18> getQ_(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
#ifdef DEBUG
                    // --- [IMPROVEMENT] Confirm the specialized override is being used ---
                    std::cout << "    -> [SINGER DEBUG | getQ_] Using SINGER's overridden getQ_ method." << std::endl;
#endif
                    return getQ(dt, alpha_diag_, Qc_diag);
                }

                Eigen::Matrix<double, 18, 18> getQinv_(const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
#ifdef DEBUG
                    // --- [IMPROVEMENT] Confirm the specialized override is being used ---
                    std::cout << "    -> [SINGER DEBUG | getQinv_] Using SINGER's overridden getQinv_ method." << std::endl;
#endif
                    return getQ(dt, alpha_diag_, Qc_diag).inverse();
                }

                Evaluable<PoseType>::Ptr getPoseInterpolator_(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                    return PoseInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
                }

                Evaluable<VelocityType>::Ptr getVelocityInterpolator_(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                    return VelocityInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
                }

                Evaluable<AccelerationType>::Ptr getAccelerationInterpolator_(const Time time, const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                    return AccelerationInterpolator::MakeShared(time, knot1, knot2,
                                                                alpha_diag_);
                }

                Evaluable<PoseType>::Ptr getPoseExtrapolator_(const Time time, const Variable::ConstPtr& knot) const {
                    return PoseExtrapolator::MakeShared(time, knot, alpha_diag_);
                }

                Evaluable<VelocityType>::Ptr getVelocityExtrapolator_(const Time time, const Variable::ConstPtr& knot) const {
                    return VelocityExtrapolator::MakeShared(time, knot, alpha_diag_);
                }

                Evaluable<AccelerationType>::Ptr getAccelerationExtrapolator_(const Time time, const Variable::ConstPtr& knot) const {
                    return AccelerationExtrapolator::MakeShared(time, knot, alpha_diag_);
                }

                Evaluable<Eigen::Matrix<double, 18, 1>>::Ptr getPriorFactor_(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
                    return PriorFactor::MakeShared(knot1, knot2, alpha_diag_);
                }
            };
        }  // namespace bspline
    }  // namespace traj
}  // namespace finalicp