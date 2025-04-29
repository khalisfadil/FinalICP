#pragma once

#include <trajectory/constvel/helper.hpp>
#include <evaluable/se3/se3statevar.hpp> 
#include <evaluable/statevar.hpp>
#include <evaluable/vspace/vspacestatevar.hpp> 
#include <problem/costterm/basecostterm.hpp>
#include <problem/costterm/p2psupercostterm.hpp>
#include <problem/costterm/imusupercostterm.hpp>
#include <problem/lossfunc/lossfunc.hpp>
#include <problem/problem.hpp>
#include <trajectory/constvel/interface.hpp>
#include <trajectory/time.hpp>

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

    struct IntegratedState {
        Eigen::Matrix3d C_mr;
        Eigen::Vector3d r_rm_in_m;
        Eigen::Vector3d v_rm_in_m;
        double timestamp;
        Eigen::Matrix<double, 6, 15> jacobian;  // [dp / d delta x; dC / d delta x], x = (pos, rot, vel, b_a, b_g)

        IntegratedState(Eigen::Matrix3d C_mr_, Eigen::Vector3d r_rm_in_m_, Eigen::Vector3d v_rm_in_m_, double timestamp_) : C_mr(C_mr_), r_rm_in_m(r_rm_in_m_), v_rm_in_m(v_rm_in_m_), timestamp(timestamp_) {
            jacobian = Eigen::Matrix<double, 6, 15>::Zero();
        }
        IntegratedState() {}
    };

    class P2PGlobalSuperCostTerm : public BaseCostTerm {

        public:

            enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

            struct Options {
                int num_threads = 1;
                LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
                double p2p_loss_sigma = 0.1;
                double r_p2p = 1.0;
                Eigen::Matrix<double, 3, 1> gravity = {0, 0, -9.8042};
            };

            using Ptr = std::shared_ptr<P2PGlobalSuperCostTerm>;
            using ConstPtr = std::shared_ptr<const P2PGlobalSuperCostTerm>;
            using PoseType = math::se3::Transformation;
            using VelType = Eigen::Matrix<double, 3, 1>;
            using BiasType = Eigen::Matrix<double, 6, 1>;
            using Time = traj::Time;

            //Factory method to create an instance of `IMUSuperCostTerm`.
            static Ptr MakeShared(const Time time, const Evaluable<PoseType>::ConstPtr &transform_r_to_m, const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m,
                                    const Evaluable<BiasType>::ConstPtr &bias,const Options &options,const std::vector<IMUData> &imu_data_vec);

            //Constructs the IMU-based cost term for trajectory optimization.
            P2PGlobalSuperCostTerm(
                const Time time,
                const Evaluable<PoseType>::ConstPtr &transform_r_to_m,
                const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m,
                const Evaluable<BiasType>::ConstPtr &bias,
                const Options &options,
                const std::vector<IMUData> &imu_data_vec)
                : time_(time),
                    transform_r_to_m_(transform_r_to_m),
                    v_m_to_r_in_m_(v_m_to_r_in_m),
                    bias_(bias),
                    options_(options),
                    curr_time_(time.seconds()) {

                p2p_loss_func_ = [this]() -> BaseLossFunc::Ptr {
                    switch (options_.p2p_loss_func) {
                        case LOSS_FUNC::L2: return L2LossFunc::MakeShared();
                        case LOSS_FUNC::DCS: return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
                        case LOSS_FUNC::CAUCHY: return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
                        case LOSS_FUNC::GM: return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
                        default:
                        return nullptr;
                    }
                    return nullptr;
                }();
                gravity_ = options_.gravity;

                for (auto imu_data : imu_data_vec) {
                    imu_data_vec_.push_back(imu_data);
                }

                for (auto imu_data : imu_data_vec_) {
                    if (imu_data.timestamp < curr_time_) {
                        imu_before.push_back(imu_data);
                    } else {
                        imu_after.push_back(imu_data);
                    }
                }

                if (imu_before.size() > 0) {
                    std::reverse(imu_before.begin(), imu_before.end());
                }
            }

            //Computes the cost contribution to the objective function.
            double cost() const override;

            //Retrieves the set of related variable keys.
            void getRelatedVarKeys(KeySet &keys) const override;

            //Initializes precomputed interpolation matrices and Jacobians.
            void initP2PMatches();

            //Appends IMU data for cost term evaluation.
            void emplace_back(P2PMatch &p2p_match) {p2p_matches_.emplace_back(p2p_match);}

            //Clears stored IMU data.
            void clear() { p2p_matches_.clear(); }

            //Reserves space for IMU data storage.
            void reserve(unsigned int N) { p2p_matches_.reserve(N); }

            //Retrieves stored IMU data.
            std::vector<P2PMatch> &get() { return p2p_matches_; }

            //Computes and accumulates Gauss-Newton terms for optimization.
            void buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const override;

            std::vector<IntegratedState> integrate_(bool compute_jacobians) const;

            void set_min_time(double min_time) {min_point_time_ = min_time;}

            void set_max_time(double max_time) {max_point_time_ = max_time;}

        private:
            const Time time_;
            const Evaluable<PoseType>::ConstPtr transform_r_to_m_;
            const Evaluable<VelType>::ConstPtr v_m_to_r_in_m_;
            const Evaluable<BiasType>::ConstPtr bias_;
            const Options options_;
            const double curr_time_;
            std::vector<IMUData> imu_data_vec_;
            std::vector<IMUData> imu_before;
            std::vector<IMUData> imu_after;
            std::vector<P2PMatch> p2p_matches_;
            std::map<double, std::vector<int>> p2p_match_bins_;
            std::vector<double> meas_times_;
            double min_point_time_ = 0;
            double max_point_time_ = 0;
            BaseLossFunc::Ptr p2p_loss_func_ = L2LossFunc::MakeShared();
            Eigen::Vector3d gravity_ = {0, 0, -9.8042};
    };
}  // namespace finalicp