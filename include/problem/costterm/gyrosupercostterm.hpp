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

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/spin_mutex.h>
#include <tbb/combinable.h>

#include <vector>
#include <cmath>
#include <atomic>
#include <stdexcept>

#include <iostream>

namespace finalicp {

    //Implements a cost term for gyro bias estimation in trajectory optimization.
    class GyroSuperCostTerm : public BaseCostTerm {
        public:
            //Defines available robust loss functions for gyro bias optimization.
            enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

            //Configuration settings for `GyroSuperCostTerm`.
            struct Options {
                int num_threads = 1;
                LOSS_FUNC gyro_loss_func = LOSS_FUNC::CAUCHY;
                double gyro_loss_sigma = 0.1;
                Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
                bool se2 = false;
            };

            using Ptr = std::shared_ptr<GyroSuperCostTerm>;
            using ConstPtr = std::shared_ptr<const GyroSuperCostTerm>;
            using PoseType = math::se3::Transformation;
            using VelType = Eigen::Matrix<double, 6, 1>;
            using BiasType = Eigen::Matrix<double, 6, 1>;
            using Interface = traj::const_vel::Interface;
            using Variable = traj::const_vel::Variable;
            using Time = traj::Time;
            using Matrix12d = Eigen::Matrix<double, 12, 12>;
            using Matrix6d = Eigen::Matrix<double, 6, 6>;

            //Factory method to create an instance of `GyroSuperCostTerm`.
            static Ptr MakeShared(const Interface::ConstPtr &interface, const Time time1,
                                    const Time time2,
                                    const Evaluable<BiasType>::ConstPtr &bias1,
                                    const Evaluable<BiasType>::ConstPtr &bias2,
                                    const Options &options);

            //Constructs the gyroscope cost term for optimization.
            GyroSuperCostTerm(const Interface::ConstPtr &interface, const Time time1,
                                const Time time2,
                                const Evaluable<BiasType>::ConstPtr &bias1,
                                const Evaluable<BiasType>::ConstPtr &bias2,
                                const Options &options)
                : interface_(interface),
                    time1_(time1),
                    time2_(time2),
                    bias1_(bias1),
                    bias2_(bias2),
                    options_(options),
                    knot1_(interface_->get(time1)),
                    knot2_(interface_->get(time2)) {
                const double T = (knot2_->time() - knot1_->time()).seconds();
                const Eigen::Matrix<double, 6, 1> ones =
                    Eigen::Matrix<double, 6, 1>::Ones();
                Qinv_T_ = traj::const_vel::getQinv(T, ones);
                Tran_T_ = traj::const_vel::getTran(T);

                gyro_loss_func_ = [this]() -> BaseLossFunc::Ptr {
                switch (options_.gyro_loss_func) {
                    case LOSS_FUNC::L2: return L2LossFunc::MakeShared();
                    case LOSS_FUNC::DCS: return DcsLossFunc::MakeShared(options_.gyro_loss_sigma);
                    case LOSS_FUNC::CAUCHY: return CauchyLossFunc::MakeShared(options_.gyro_loss_sigma);
                    case LOSS_FUNC::GM: return GemanMcClureLossFunc::MakeShared(options_.gyro_loss_sigma);
                    default:
                    return nullptr;
                }
                return nullptr;
                }();

                jac_vel_.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
                jac_bias_.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity() * -1;

                if (options_.se2) {
                jac_vel_(0, 5) = 1;
                jac_bias_(0, 5) = -1;
                }

                Eigen::Matrix3d R_gyro = Eigen::Matrix3d::Zero();
                R_gyro.diagonal() = options_.r_imu_ang;
                gyro_noise_model_ = StaticNoiseModel<3>::MakeShared(R_gyro);
            }

            //Computes the cost contribution to the objective function.
            double cost() const override;

            //Retrieves the set of related variable keys.
            void getRelatedVarKeys(KeySet &keys) const override;

            //Initializes precomputed interpolation matrices and Jacobians.
            void init();

            //Appends IMU data for cost term evaluation.
            void emplace_back(IMUData &imu_data) { imu_data_vec_.emplace_back(imu_data); }

            //Clears stored IMU data.
            void clear() { imu_data_vec_.clear(); }

            //Reserves space for IMU data storage.
            void reserve(unsigned int N) { imu_data_vec_.reserve(N); }

            //Retrieves stored IMU data.
            std::vector<IMUData> &get() { return imu_data_vec_; }

            //Sets IMU data for processing.
            void set(const std::vector<IMUData> imu_data_vec) {
                imu_data_vec_ = imu_data_vec;
            }

            //Computes and accumulates Gauss-Newton terms for optimization.
            void buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const override;

        private:
            const Interface::ConstPtr interface_;                                               //Pointer to the trajectory interface containing state variables.
            const Time time1_;                                                                  //Start and end times for gyroscope and accelerometer bias estimation. 
            const Time time2_;                                                                  //Start and end times for gyroscope and accelerometer bias estimation.
            const Evaluable<BiasType>::ConstPtr bias1_;                                         //Bias state variables at `time1_` and `time2_`.
            const Evaluable<BiasType>::ConstPtr bias2_;                                         //Bias state variables at `time1_` and `time2_`.
            const Options options_;                                                             //Configuration options for loss functions and bias estimation settings. */
            const Variable::ConstPtr knot1_;                                                    //Trajectory knots corresponding to `time1_` and `time2_`.
            const Variable::ConstPtr knot2_;                                                    //Trajectory knots corresponding to `time1_` and `time2_`.
            Matrix12d Qinv_T_ = Matrix12d::Identity();                                          //Precomputed inverse covariance matrix for process noise.
            Matrix12d Tran_T_ = Matrix12d::Identity();                                          //Precomputed inverse covariance matrix for process noise.
            std::map<double, std::pair<Eigen::Matrix4d, Eigen::Matrix4d>> interp_mats_;         //Precomputed interpolation matrices for IMU data alignment.
            std::vector<IMUData> imu_data_vec_;                                                 //Container storing raw IMU data measurements.
            std::vector<double> meas_times_;                                                    //Stores measurement timestamps for IMU readings.
            BaseLossFunc::Ptr gyro_loss_func_ = L2LossFunc::MakeShared();                       //Loss function for robust gyroscope bias optimization.
            const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();              
            StaticNoiseModel<3>::Ptr gyro_noise_model_ = StaticNoiseModel<3>::MakeShared(R);    //Noise model for gyroscope bias estimation.
            Eigen::Matrix<double, 3, 6> jac_vel_ = Eigen::Matrix<double, 3, 6>::Zero();         //Jacobian matrix for velocity propagation.s
            Eigen::Matrix<double, 3, 6> jac_bias_ = Eigen::Matrix<double, 3, 6>::Zero();        //Jacobian matrix for gyroscope bias propagation.
            void initialize_interp_matrices_();                                                 //nitializes precomputed interpolation matrices.
    };
}  // namespace finalicp