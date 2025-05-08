#pragma once

#include <trajectory/constvel/helper.hpp>
#include <evaluable/se3/se3statevar.hpp> 
#include <evaluable/statevar.hpp>
#include <evaluable/vspace/vspacestatevar.hpp> 
#include <problem/costterm/basecostterm.hpp>
#include <problem/costterm/imusupercostterm.hpp>
#include <problem/lossfunc/lossfunc.hpp>
#include <problem/noisemodel/staticnoisemodel.hpp>
#include <problem/problem.hpp>
#include <trajectory/constvel/interface.hpp>
#include <trajectory/time.hpp>

#include <vector>
#include <cmath>
#include <atomic>
#include <stdexcept>

#include <iostream>

namespace finalicp {

    class PreintAccCostTerm : public BaseCostTerm {

        public:

            enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

            struct Options {
                int num_threads = 1;
                LOSS_FUNC loss_func = LOSS_FUNC::L2;
                double loss_sigma = 1.0;
                Eigen::Matrix<double, 3, 1> gravity = Eigen::Matrix<double, 3, 1>::Zero();
                Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
                bool se2 = false;
            };

            using Ptr = std::shared_ptr<PreintAccCostTerm>;
            using ConstPtr = std::shared_ptr<const PreintAccCostTerm>;
            using PoseType = math::se3::Transformation;
            using VelType = Eigen::Matrix<double, 6, 1>;
            using BiasType = Eigen::Matrix<double, 6, 1>;
            using Interface = traj::const_vel::Interface;
            using Variable = traj::const_vel::Variable;
            using Time = traj::Time;
            using Matrix12d = Eigen::Matrix<double, 12, 12>;
            using Matrix6d = Eigen::Matrix<double, 6, 6>;

            //Factory method to create an instance of `IMUSuperCostTerm`.
            static Ptr MakeShared(const Interface::ConstPtr &interface, const Time time1,
                        const Time time2,
                        const Evaluable<BiasType>::ConstPtr &bias1,
                        const Evaluable<BiasType>::ConstPtr &bias2,
                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
                        const Options &options);

            //Constructs the IMU-based cost term for trajectory optimization.
            PreintAccCostTerm(const Interface::ConstPtr &interface, const Time time1,
                                const Time time2,
                                const Evaluable<BiasType>::ConstPtr &bias1,
                                const Evaluable<BiasType>::ConstPtr &bias2,
                                const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
                                const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
                                const Options &options)
                : interface_(interface),
                    time1_(time1),
                    time2_(time2),
                    bias1_(bias1),
                    bias2_(bias2),
                    transform_i_to_m_1_(transform_i_to_m_1),
                    transform_i_to_m_2_(transform_i_to_m_2),
                    options_(options),
                    knot1_(interface_->get(time1)),
                    knot2_(interface_->get(time2)) {
                const double T = (knot2_->time() - knot1_->time()).seconds();
                const Eigen::Matrix<double, 6, 1> ones =
                    Eigen::Matrix<double, 6, 1>::Ones();
                Qinv_T_ = traj::const_vel::getQinv(T, ones);
                Tran_T_ = traj::const_vel::getTran(T);

                loss_func_ = [this]() -> BaseLossFunc::Ptr {
                switch (options_.loss_func) {
                    case LOSS_FUNC::L2:
                    return L2LossFunc::MakeShared();
                    case LOSS_FUNC::DCS:
                    return DcsLossFunc::MakeShared(options_.loss_sigma);
                    case LOSS_FUNC::CAUCHY:
                    return CauchyLossFunc::MakeShared(options_.loss_sigma);
                    case LOSS_FUNC::GM:
                    return GemanMcClureLossFunc::MakeShared(options_.loss_sigma);
                    default:
                    return nullptr;
                }
                return nullptr;
                }();

                jac_bias_accel_.block<3, 3>(0, 0) =
                    Eigen::Matrix<double, 3, 3>::Identity() * -1;

                if (options_.se2) {
                jac_bias_accel_(2, 2) = 0.;
                }

                R_acc_.diagonal() = options_.r_imu_acc;
            }

            //Computes the cost contribution to the objective function.
            double cost() const override;

            //Retrieves the set of related variable keys.
            void getRelatedVarKeys(KeySet &keys) const override;

            void init();

            //Appends IMU data for cost term evaluation.
            void emplace_back(IMUData &imu_data) { imu_data_vec_.emplace_back(imu_data); }

            //Clears stored IMU data.
            void clear() { imu_data_vec_.clear(); }

            //Reserves space for IMU data storage.
            void reserve(unsigned int N) { imu_data_vec_.reserve(N); }

            //Retrieves stored IMU data.
            std::vector<IMUData> &get() { return imu_data_vec_; }

            void set(const std::vector<IMUData> imu_data_vec) {imu_data_vec_ = imu_data_vec;}

            //Computes and accumulates Gauss-Newton terms for optimization.
            void buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const override;

        private:
            const Interface::ConstPtr interface_;
            const Time time1_;
            const Time time2_;
            const Evaluable<BiasType>::ConstPtr bias1_;
            const Evaluable<BiasType>::ConstPtr bias2_;
            const Evaluable<PoseType>::ConstPtr transform_i_to_m_1_;
            const Evaluable<PoseType>::ConstPtr transform_i_to_m_2_;
            const Options options_;
            const Variable::ConstPtr knot1_;
            const Variable::ConstPtr knot2_;
            Matrix12d Qinv_T_ = Matrix12d::Identity();
            Matrix12d Tran_T_ = Matrix12d::Identity();
            std::map<double, std::pair<Eigen::Matrix4d, Eigen::Matrix4d>> interp_mats_;
            std::vector<IMUData> imu_data_vec_;
            std::vector<double> meas_times_;
            BaseLossFunc::Ptr loss_func_ = L2LossFunc::MakeShared();
            const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
            StaticNoiseModel<3>::Ptr acc_noise_model_ = StaticNoiseModel<3>::MakeShared(R);
            Eigen::Matrix<double, 3, 6> jac_bias_accel_ = Eigen::Matrix<double, 3, 6>::Zero();
            Eigen::Matrix3d R_acc_ = Eigen::Matrix3d::Zero();
            void initialize_interp_matrices_();
    };
}  // namespace finalicp