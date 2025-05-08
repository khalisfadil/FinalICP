#pragma once

#include <evaluable/se3/se3statevar.hpp> 
#include <evaluable/statevar.hpp>
#include <evaluable/vspace/vspacestatevar.hpp> 
#include <problem/costterm/basecostterm.hpp>
#include <problem/lossfunc/lossfunc.hpp>
#include <problem/noisemodel/staticnoisemodel.hpp>
#include <problem/problem.hpp>
#include <trajectory/constacc/interface.hpp> 
#include <trajectory/time.hpp>

#include <vector>
#include <cmath>
#include <atomic>
#include <stdexcept>

#include <iostream>

namespace finalicp {

    //Holds raw IMU measurement data.
    struct IMUData {
        double timestamp = 0;
        Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d lin_acc = Eigen::Vector3d::Zero();

        IMUData(double timestamp_, Eigen::Vector3d ang_vel_, Eigen::Vector3d lin_acc_)
            : timestamp(timestamp_), ang_vel(ang_vel_), lin_acc(lin_acc_) {}

        IMUData() {}
    };

    class IMUSuperCostTerm : public BaseCostTerm {
        public:
            enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

            struct Options {
                int num_threads = 1;
                LOSS_FUNC acc_loss_func = LOSS_FUNC::CAUCHY;
                LOSS_FUNC gyro_loss_func = LOSS_FUNC::CAUCHY;
                double acc_loss_sigma = 0.1;
                double gyro_loss_sigma = 0.1;
                Eigen::Matrix<double, 3, 1> gravity = Eigen::Matrix<double, 3, 1>::Zero();
                Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
                Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
                bool se2 = false;
                bool use_accel = true;
            };

            using Ptr = std::shared_ptr<IMUSuperCostTerm>;
            using ConstPtr = std::shared_ptr<const IMUSuperCostTerm>;
            using PoseType = math::se3::Transformation;
            using VelType = Eigen::Matrix<double, 6, 1>;
            using AccType = Eigen::Matrix<double, 6, 1>;
            using BiasType = Eigen::Matrix<double, 6, 1>;
            using Interface = traj::const_acc::Interface;
            using Variable = traj::const_acc::Variable;
            using Time = traj::Time;
            using Matrix18d = Eigen::Matrix<double, 18, 18>;
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
            IMUSuperCostTerm(const Interface::ConstPtr &interface, const Time time1,
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
                Qinv_T_ = interface_->getQinvPublic(T, ones);
                Tran_T_ = interface_->getTranPublic(T);

                acc_loss_func_ = [this]() -> BaseLossFunc::Ptr {
                    switch (options_.acc_loss_func) {
                        case LOSS_FUNC::L2: return L2LossFunc::MakeShared();
                        case LOSS_FUNC::DCS: return DcsLossFunc::MakeShared(options_.acc_loss_sigma);
                        case LOSS_FUNC::CAUCHY: return CauchyLossFunc::MakeShared(options_.acc_loss_sigma);
                        case LOSS_FUNC::GM: return GemanMcClureLossFunc::MakeShared(options_.acc_loss_sigma);
                        default:
                        return nullptr;
                    }
                    return nullptr;
                }();

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
                jac_accel_.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();
                jac_bias_accel_.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * -1;
                jac_bias_gyro_.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity() * -1;

                if (options_.se2) {
                    jac_vel_.block<2, 2>(1, 4).setZero();
                    jac_accel_(2, 2) = 0.;
                    jac_bias_accel_(2, 2) = 0.;
                    jac_bias_gyro_.block<2, 2>(1, 4).setZero();
                }

                Eigen::Matrix3d R_acc = Eigen::Matrix3d::Zero();
                R_acc.diagonal() = options_.r_imu_acc;
                Eigen::Matrix3d R_gyro = Eigen::Matrix3d::Zero();
                R_gyro.diagonal() = options_.r_imu_ang;
                acc_noise_model_ = StaticNoiseModel<3>::MakeShared(R_acc);
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

            const Interface::ConstPtr interface_;                                                   //Shared pointer to the trajectory interface.
            const Time time1_;                                                                      //Start and end times for gyroscope and accelerometer bias estimation.
            const Time time2_;                                                                      //Start and end times for gyroscope and accelerometer bias estimation.
            const Evaluable<BiasType>::ConstPtr bias1_;                                             //Bias state variables at `time1_` and `time2_`.
            const Evaluable<BiasType>::ConstPtr bias2_;                                             //Bias state variables at `time1_` and `time2_`.
            const Evaluable<PoseType>::ConstPtr transform_i_to_m_1_;                                //Transformation matrices from IMU to world frame.
            const Evaluable<PoseType>::ConstPtr transform_i_to_m_2_;                                //Transformation matrices from IMU to world frame.
            const Options options_;                                                                 //Configuration settings for loss functions, bias estimation, and sensor parameters. */
            const Variable::ConstPtr knot1_;                                                        //Trajectory knots corresponding to `time1_` and `time2_`.
            const Variable::ConstPtr knot2_;                                                        //Trajectory knots corresponding to `time1_` and `time2_`.
            Matrix18d Qinv_T_ = Matrix18d::Identity();                                              //Precomputed inverse covariance matrix for process noise.
            Matrix18d Tran_T_ = Matrix18d::Identity();                                              //Precomputed inverse covariance matrix for process noise.
            std::map<double, std::pair<Eigen::Matrix3d, Eigen::Matrix3d>> interp_mats_;             //Precomputed interpolation matrices for IMU data alignment.
            std::vector<IMUData> imu_data_vec_;                                                     //Stores raw IMU data (gyroscope and accelerometer readings).
            std::vector<double> meas_times_;                                                        //Stores measurement timestamps for IMU readings.
            BaseLossFunc::Ptr acc_loss_func_ = L2LossFunc::MakeShared();                            //Loss function for accelerometer bias estimation.
            BaseLossFunc::Ptr gyro_loss_func_ = L2LossFunc::MakeShared();                           //Loss function for gyroscope bias estimation.
            const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();              
            StaticNoiseModel<3>::Ptr acc_noise_model_ = StaticNoiseModel<3>::MakeShared(R);         //Noise model for accelerometer bias estimation.
            StaticNoiseModel<3>::Ptr gyro_noise_model_ = StaticNoiseModel<3>::MakeShared(R);        //Noise model for gyroscope bias estimation.
            Eigen::Matrix<double, 3, 6> jac_vel_ = Eigen::Matrix<double, 3, 6>::Zero();             //Jacobian matrix mapping velocity updates.
            Eigen::Matrix<double, 3, 6> jac_accel_ = Eigen::Matrix<double, 3, 6>::Zero();           //Jacobian matrix mapping accelerometer updates.
            Eigen::Matrix<double, 3, 6> jac_bias_accel_ = Eigen::Matrix<double, 3, 6>::Zero();      //Jacobian matrix mapping accelerometer bias to errors.
            Eigen::Matrix<double, 3, 6> jac_bias_gyro_ =Eigen::Matrix<double, 3, 6>::Zero();        //Jacobian matrix mapping gyroscope bias to errors.
            void initialize_interp_matrices_();                                                     //Initializes precomputed interpolation matrices.
    };

    

}  // namespace finalicp