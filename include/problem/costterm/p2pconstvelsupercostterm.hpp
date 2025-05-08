#pragma once

#include <trajectory/constvel/helper.hpp>
#include <evaluable/se3/se3statevar.hpp> 
#include <evaluable/statevar.hpp>
#include <evaluable/vspace/vspacestatevar.hpp> 
#include <problem/costterm/basecostterm.hpp>
#include <problem/costterm/p2psupercostterm.hpp>
#include <problem/lossfunc/lossfunc.hpp>
#include <problem/problem.hpp>
#include <trajectory/constvel/interface.hpp>
#include <trajectory/time.hpp>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
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

    class P2PCVSuperCostTerm : public BaseCostTerm {

        public:

            enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

            struct Options {
                int num_threads = 1;
                LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
                double p2p_loss_sigma = 0.1;
                double r_p2p = 1.0;
            };

            using Ptr = std::shared_ptr<P2PCVSuperCostTerm>;
            using ConstPtr = std::shared_ptr<const P2PCVSuperCostTerm>;
            using PoseType = math::se3::Transformation;
            using VelType = Eigen::Matrix<double, 6, 1>;
            using Interface = traj::const_vel::Interface;
            using Variable = traj::const_vel::Variable;
            using Time = traj::Time;
            using Matrix12d = Eigen::Matrix<double, 12, 12>;
            using Matrix6d = Eigen::Matrix<double, 6, 6>;

            //Factory method to create an instance of `IMUSuperCostTerm`.
            static Ptr MakeShared(const Interface::ConstPtr &interface, const Time time1, const Time time2, const Options &options);

            //Constructs the IMU-based cost term for trajectory optimization.
            P2PCVSuperCostTerm(const Interface::ConstPtr &interface, const Time time1, const Time time2, const Options &options)
                : interface_(interface),
                    time1_(time1),
                    time2_(time2),
                    options_(options),
                    knot1_(interface_->get(time1)),
                    knot2_(interface_->get(time2)) {
                const double T = (knot2_->time() - knot1_->time()).seconds();
                const Eigen::Matrix<double, 6, 1> ones =
                    Eigen::Matrix<double, 6, 1>::Ones();
                Qinv_T_ = traj::const_vel::getQinv(T, ones);
                Tran_T_ = traj::const_vel::getTran(T);

                p2p_loss_func_ = [this]() -> BaseLossFunc::Ptr {
                    switch (options_.p2p_loss_func) {
                        case LOSS_FUNC::L2: return L2LossFunc::MakeShared();
                        case LOSS_FUNC::DCS: return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
                        case LOSS_FUNC::CAUCHY:return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
                        case LOSS_FUNC::GM: return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
                        default:
                        return nullptr;
                    }
                    return nullptr;
                }();
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

        private:
            const Interface::ConstPtr interface_;                                               //Shared pointer to the trajectory interface.
            const Time time1_;                                                                  //Start and end times for gyroscope and accelerometer bias estimation.
            const Time time2_;                                                                  //Start and end times for gyroscope and accelerometer bias estimation.
            const Options options_;                                                             //Bias state variables at `time1_` and `time2_`.
            const Variable::ConstPtr knot1_;                                                    //Trajectory knots corresponding to `time1_` and `time2_`.
            const Variable::ConstPtr knot2_;                                                    //Trajectory knots corresponding to `time1_` and `time2_`.
            Matrix12d Qinv_T_ = Matrix12d::Identity();                                          //Precomputed inverse covariance matrix for process noise.
            Matrix12d Tran_T_ = Matrix12d::Identity();                                          //Precomputed inverse covariance matrix for process noise.
            std::map<double, std::pair<Eigen::Matrix4d, Eigen::Matrix4d>> interp_mats_;         //Precomputed interpolation matrices for IMU data alignment.
            std::vector<P2PMatch> p2p_matches_;
            std::map<double, std::vector<int>> p2p_match_bins_;
            std::vector<double> meas_times_;                                                    //Stores measurement timestamps for IMU readings.
            BaseLossFunc::Ptr p2p_loss_func_ = L2LossFunc::MakeShared();
            void initialize_interp_matrices_();                                                 //Initializes precomputed interpolation matrices.
    };
}  // namespace finalicp