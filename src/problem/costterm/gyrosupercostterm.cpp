#include <problem/costterm/gyrosupercostterm.hpp>

#include <iostream>

namespace finalicp {

    GyroSuperCostTerm::Ptr GyroSuperCostTerm::MakeShared(
        const Interface::ConstPtr &interface, const Time time1, const Time time2,
        const Evaluable<BiasType>::ConstPtr &bias1,
        const Evaluable<BiasType>::ConstPtr &bias2, const Options &options) {
        return std::make_shared<GyroSuperCostTerm>(interface, time1, time2, bias1,bias2, options);
    }


    double GyroSuperCostTerm::cost() const {
        // Initialize accumulators (thread-safe for parallel case)
        std::atomic<double> total_cost{0.0};
        std::atomic<size_t> nan_count{0};
        std::atomic<size_t> exception_count{0};

        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();
        const auto b1_ = bias1_->forward();
        const auto b2_ = bias2_->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();
        const auto b1 = b1_->value();
        const auto b2 = b2_->value();

        // Compute relative pose and velocity transformation
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto J_21_inv_w2 = J_21_inv * w2;

        // Sequential processing for small imu_data_vec_ to avoid parallel overhead
        if (imu_data_vec_.size() < 100) { // Tune threshold via profiling
            double cost = 0;
            for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
                try {
                    const double &ts = imu_data_vec_[i].timestamp;
                    const IMUData &imu_data = imu_data_vec_[i];

                    // Pose interpolation
                    const auto& omega = interp_mats_.at(ts).first;
                    const auto& lambda = interp_mats_.at(ts).second;

                    const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                    const Eigen::Matrix<double, 6, 1> xi_j1 =lambda(1, 1) * w1 + omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2;
                    const Eigen::Matrix<double, 6, 1> w_i = math::se3::vec2jac(xi_i1) * xi_j1;

                    // Interpolated bias
                    Eigen::Matrix<double, 6, 1> bias_i = Eigen::Matrix<double, 6, 1>::Zero();
                    const double tau = ts - knot1_->time().seconds();
                    const double T = knot2_->time().seconds() - knot1_->time().seconds();
                    const double ratio = tau / T;
                    const double omega_ = ratio;
                    const double lambda_ = 1 - ratio;
                    bias_i = lambda_ * b1 + omega_ * b2;

                    // Compute gyroscope error
                    Eigen::Matrix<double, 3, 1> raw_error_gyro = Eigen::Matrix<double, 3, 1>::Zero();
                    if (options_.se2) {
                        raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                    } else {
                        raw_error_gyro = imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                    }

                    // Evaluate cost
                    double cost_i = gyro_loss_func_->cost(
                        gyro_noise_model_->getWhitenedErrorNorm(raw_error_gyro));
                    if (std::isnan(cost_i)) {
                        ++nan_count;
                    } else {
                        cost += cost_i;
                    }

                } catch (const std::exception& e) {
                    ++exception_count;
                    std::cerr << "[GyroSuperCostTerm::cost] exception at timestamp " << imu_data_vec_[i].timestamp << ": " << e.what() << std::endl;
                } catch (...) {
                    ++exception_count;
                    std::cerr << "[GyroSuperCostTerm::cost] exception at timestamp " << imu_data_vec_[i].timestamp << ": (unknown)" << std::endl;
                }
            }
            if (nan_count > 0) {
                std::cerr << "[GyroSuperCostTerm::cost] Warning: " << nan_count << " NaN cost terms ignored!" << std::endl;
            }
            if (exception_count > 0) {
                std::cerr << "[GyroSuperCostTerm::cost] Warning: " << exception_count << " exceptions occurred!" << std::endl;
            }
            return cost;
        }

        // Parallel processing with TBB parallel_for for large imu_data_vec_
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, options_.num_threads);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, imu_data_vec_.size(), 100),
            [&total_cost, &nan_count, &exception_count, &w1, &xi_21, &J_21_inv_w2, &b1, &b2,
            this](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    try {

                        const double &ts = imu_data_vec_[i].timestamp;
                        const IMUData &imu_data = imu_data_vec_[i];

                        // Pose interpolation
                        const auto& omega = interp_mats_.at(ts).first;
                        const auto& lambda = interp_mats_.at(ts).second;
                        const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                        const Eigen::Matrix<double, 6, 1> xi_j1 = lambda(1, 1) * w1 + omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2;
                        const Eigen::Matrix<double, 6, 1> w_i = math::se3::vec2jac(xi_i1) * xi_j1;

                        // Interpolated bias
                        Eigen::Matrix<double, 6, 1> bias_i = Eigen::Matrix<double, 6, 1>::Zero();
                        const double tau = ts - knot1_->time().seconds();
                        const double T = knot2_->time().seconds() - knot1_->time().seconds();
                        const double ratio = tau / T;
                        const double omega_ = ratio;
                        const double lambda_ = 1 - ratio;
                        bias_i = lambda_ * b1 + omega_ * b2;

                        // Compute gyroscope error
                        Eigen::Matrix<double, 3, 1> raw_error_gyro = Eigen::Matrix<double, 3, 1>::Zero();
                        if (options_.se2) {
                            raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                        } else {
                            raw_error_gyro = imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                        }

                        // Evaluate cost
                        double cost_i = gyro_loss_func_->cost(
                            gyro_noise_model_->getWhitenedErrorNorm(raw_error_gyro));
                        if (std::isnan(cost_i)) {
                            ++nan_count; // Atomic increment
                        } else {
                            total_cost += cost_i; // Atomic addition
                        }

                    } catch (const std::exception& e) {
                        ++exception_count; // Atomic increment
                        std::cerr << "[GyroSuperCostTerm::cost] STEAM exception at index " << i
                                << ", timestamp " << imu_data_vec_[i].timestamp << ": " << e.what() << std::endl;
                    } catch (...) {
                        ++exception_count; // Atomic increment
                        std::cerr << "[GyroSuperCostTerm::cost] STEAM exception at index " << i
                                << ", timestamp " << imu_data_vec_[i].timestamp << ": (unknown)" << std::endl;
                    }
                }
            }
        );

        // Log warnings after parallel processing
        if (nan_count > 0) {
            std::cerr << "[GyroSuperCostTerm::cost] Warning: " << nan_count << " NaN cost terms ignored!" << std::endl;
        }
        if (exception_count > 0) {
            std::cerr << "[GyroSuperCostTerm::cost] Warning: " << exception_count << " exceptions occurred!" << std::endl;
        }

        return total_cost;
    }

    void GyroSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
        knot1_->pose()->getRelatedVarKeys(keys);
        knot2_->pose()->getRelatedVarKeys(keys);
        knot1_->velocity()->getRelatedVarKeys(keys);
        knot2_->velocity()->getRelatedVarKeys(keys);
        bias1_->getRelatedVarKeys(keys);
        bias2_->getRelatedVarKeys(keys);
    }

    void GyroSuperCostTerm::init() { initialize_interp_matrices_(); }

    void GyroSuperCostTerm::initialize_interp_matrices_() {
        const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
        // #pragma omp parallel for num_threads(options_.num_threads)
        for (const IMUData &imu_data : imu_data_vec_) {
            const auto time = imu_data.timestamp;
            if (interp_mats_.find(time) == interp_mats_.end()) {
            const double tau = (Time(time) - time1_).seconds();
            // const double T = (time2_ - time1_).seconds();
            // const double ratio = tau / T;
            // const double ratio2 = ratio * ratio;
            // const double ratio3 = ratio2 * ratio;
            // Calculate 'omega' interpolation values
            Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
            // omega(0, 0) = 3.0 * ratio2 - 2.0 * ratio3;
            // omega(0, 1) = tau * (ratio2 - ratio);
            // omega(1, 0) = 6.0 * (ratio - ratio2) / T;
            // omega(1, 1) = 3.0 * ratio2 - 2.0 * ratio;
            // Calculate 'lambda' interpolation values
            Eigen::Matrix4d lambda = Eigen::Matrix4d::Zero();
            // lambda(0, 0) = 1.0 - omega(0, 0);
            // lambda(0, 1) = tau - T * omega(0, 0) - omega(0, 1);
            // lambda(1, 0) = -omega(1, 0);
            // lambda(1, 1) = 1.0 - T * omega(1, 0) - omega(1, 1);
            const double kappa = knot2_->time().seconds() - time;
            const Matrix12d Q_tau = traj::const_vel::getQ(tau, ones);
            const Matrix12d Tran_kappa = traj::const_vel::getTran(kappa);
            const Matrix12d Tran_tau = traj::const_vel::getTran(tau);
            const Matrix12d omega12 = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
            const Matrix12d lambda12 = (Tran_tau - omega12 * Tran_T_);
            omega(0, 0) = omega12(0, 0);
            omega(1, 0) = omega12(6, 0);
            omega(0, 1) = omega12(0, 6);
            omega(1, 1) = omega12(6, 6);
            lambda(0, 0) = lambda12(0, 0);
            lambda(1, 0) = lambda12(6, 0);
            lambda(0, 1) = lambda12(0, 6);
            lambda(1, 1) = lambda12(6, 6);
            
            interp_mats_.emplace(time, std::make_pair(omega, lambda));
            }
        }
    }

    void GyroSuperCostTerm::buildGaussNewtonTerms(const StateVector& state_vec, BlockSparseMatrix* approximate_hessian, BlockVector* gradient_vector) const {

        // Initialize accumulators for exceptions
        std::atomic<size_t> exception_count{0};

        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();
        const auto b1_ = bias1_->forward();
        const auto b2_ = bias2_->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();
        const auto b1 = b1_->value();
        const auto b2 = b2_->value();

        // Compute relative pose and velocity transformation
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const auto Ad_T_21 = math::se3::tranAd(T_21.matrix());
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto w2_j_21_inv = 0.5 * math::se3::curlyhat(w2) * J_21_inv;
        const auto J_21_inv_w2 = J_21_inv * w2;

        // Thread-local accumulators for A and b
        using Matrix36x36 = Eigen::Matrix<double, 36, 36>;
        using Vector36 = Eigen::Matrix<double, 36, 1>;
        tbb::combinable<Matrix36x36> local_A([]() { return Matrix36x36::Zero(); });
        tbb::combinable<Vector36> local_b([]() { return Vector36::Zero(); });

        // Process IMU data: sequential for small sizes, parallel for large
        if (imu_data_vec_.size() < 100) { // Tune threshold via profiling
            Matrix36x36 A = Matrix36x36::Zero();
            Vector36 b = Vector36::Zero();
            for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
                try {
                    const double &ts = imu_data_vec_[i].timestamp;
                    const IMUData &imu_data = imu_data_vec_[i];

                    // Interpolation
                    const auto& omega = interp_mats_.at(ts).first;
                    const auto& lambda = interp_mats_.at(ts).second;
                    const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                    const Eigen::Matrix<double, 6, 1> xi_j1 = lambda(1, 1) * w1 + omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2;
                    const Eigen::Matrix<double, 6, 1> w_i = math::se3::vec2jac(xi_i1) * xi_j1;

                    // Interpolated bias
                    Eigen::Matrix<double, 6, 1> bias_i = Eigen::Matrix<double, 6, 1>::Zero();
                    Eigen::Matrix<double, 6, 12> interp_jac_bias = Eigen::Matrix<double, 6, 12>::Zero();
                    const double tau = ts - knot1_->time().seconds();
                    const double T = knot2_->time().seconds() - knot1_->time().seconds();
                    const double ratio = tau / T;
                    const double omega_ = ratio;
                    const double lambda_ = 1 - ratio;
                    bias_i = lambda_ * b1 + omega_ * b2;
                    interp_jac_bias.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity() * lambda_;
                    interp_jac_bias.block<6, 6>(0, 6) = Eigen::Matrix<double, 6, 6>::Identity() * omega_;

                    // Velocity interpolation Jacobians
                    Eigen::Matrix<double, 6, 24> interp_jac_vel = Eigen::Matrix<double, 6, 24>::Zero();
                    const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                    const Eigen::Matrix<double, 6, 6> xi_j1_ch = -0.5 * math::se3::curlyhat(xi_j1);
                    Eigen::Matrix<double, 6, 6> w = J_i1 * (omega(1, 0) * J_21_inv + omega(1, 1) * w2_j_21_inv) + xi_j1_ch * (omega(0, 0) * J_21_inv + omega(0, 1) * w2_j_21_inv);
                    interp_jac_vel.block<6, 6>(0, 0) = -w * Ad_T_21; // T1
                    interp_jac_vel.block<6, 6>(0, 6) = (lambda(1, 1) * J_i1 + lambda(0, 1) * xi_j1_ch); // w1
                    interp_jac_vel.block<6, 6>(0, 12) = w; // T2
                    interp_jac_vel.block<6, 6>(0, 18) = omega(1, 1) * J_i1 * J_21_inv + omega(0, 1) * xi_j1_ch * J_21_inv; // w2

                    // Evaluate, weight, whiten error
                    Eigen::Matrix<double, 3, 1> raw_error_gyro = Eigen::Matrix<double, 3, 1>::Zero();
                    if (options_.se2) {
                        raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                    } else {
                        raw_error_gyro = imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                    }
                    const Eigen::Matrix<double, 3, 1> white_error_gyro = gyro_noise_model_->whitenError(raw_error_gyro);
                    const double sqrt_w_gyro = sqrt(gyro_loss_func_->weight(white_error_gyro.norm()));
                    const Eigen::Matrix<double, 3, 1> error_gyro = sqrt_w_gyro * white_error_gyro;

                    // Get Jacobians
                    Eigen::Matrix<double, 3, 36> G = Eigen::Matrix<double, 3, 36>::Zero();
                    G.block<3, 24>(0, 0) = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() * jac_vel_ * interp_jac_vel;
                    G.block<3, 12>(0, 24) = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() * jac_bias_ * interp_jac_bias;

                    // Accumulate contributions
                    A += G.transpose() * G;
                    b += (-1) * G.transpose() * error_gyro;
                } catch (const std::exception& e) {
                    ++exception_count;
                    std::cerr << "[GyroSuperCostTerm::buildGaussNewtonTerms] exception at timestamp "
                            << imu_data_vec_[i].timestamp  << ": " << e.what() << std::endl;
                } catch (...) {
                    ++exception_count;
                    std::cerr << "[GyroSuperCostTerm::buildGaussNewtonTerms] exception at timestamp "
                            << imu_data_vec_[i].timestamp  << ": (unknown)" << std::endl;
                }
            }
            local_A.local() = A;
            local_b.local() = b;
        } else {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, imu_data_vec_.size(), 100),
                [&w1, &xi_21, &J_21_inv_w2, &b1, &b2, &J_21_inv, &w2_j_21_inv, &Ad_T_21,
                    &local_A, &local_b, &exception_count, this](
                    const tbb::blocked_range<size_t>& range) {
                    auto& A = local_A.local();
                    auto& b = local_b.local();
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        try {
                            const auto& imu_data = imu_data_vec_[i];
                            const double ts = imu_data.timestamp;

                            // Interpolation
                            const auto& omega = interp_mats_.at(ts).first;
                            const auto& lambda = interp_mats_.at(ts).second;
                            const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                            const Eigen::Matrix<double, 6, 1> xi_j1 = lambda(1, 1) * w1 + omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2;
                            const Eigen::Matrix<double, 6, 1> w_i = math::se3::vec2jac(xi_i1) * xi_j1;

                            // Interpolated bias
                            Eigen::Matrix<double, 6, 1> bias_i = Eigen::Matrix<double, 6, 1>::Zero();
                            Eigen::Matrix<double, 6, 12> interp_jac_bias = Eigen::Matrix<double, 6, 12>::Zero();
                            const double tau = ts - knot1_->time().seconds();
                            const double T = knot2_->time().seconds() - knot1_->time().seconds();
                            const double ratio = tau / T;
                            const double omega_ = ratio;
                            const double lambda_ = 1 - ratio;
                            bias_i = lambda_ * b1 + omega_ * b2;
                            interp_jac_bias.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity() * lambda_;
                            interp_jac_bias.block<6, 6>(0, 6) = Eigen::Matrix<double, 6, 6>::Identity() * omega_;

                            // Velocity interpolation Jacobians
                            Eigen::Matrix<double, 6, 24> interp_jac_vel = Eigen::Matrix<double, 6, 24>::Zero();
                            const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                            const Eigen::Matrix<double, 6, 6> xi_j1_ch = -0.5 * math::se3::curlyhat(xi_j1);
                            Eigen::Matrix<double, 6, 6> w =
                                J_i1 * (omega(1, 0) * J_21_inv + omega(1, 1) * w2_j_21_inv) +
                                xi_j1_ch * (omega(0, 0) * J_21_inv + omega(0, 1) * w2_j_21_inv);
                            interp_jac_vel.block<6, 6>(0, 0) = -w * Ad_T_21; // T1
                            interp_jac_vel.block<6, 6>(0, 6) = (lambda(1, 1) * J_i1 + lambda(0, 1) * xi_j1_ch); // w1
                            interp_jac_vel.block<6, 6>(0, 12) = w; // T2
                            interp_jac_vel.block<6, 6>(0, 18) =
                                omega(1, 1) * J_i1 * J_21_inv + omega(0, 1) * xi_j1_ch * J_21_inv; // w2

                            // Evaluate, weight, whiten error
                            Eigen::Matrix<double, 3, 1> raw_error_gyro = Eigen::Matrix<double, 3, 1>::Zero();
                            if (options_.se2) {
                                raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                            } else {
                                raw_error_gyro = imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                            }
                            const Eigen::Matrix<double, 3, 1> white_error_gyro =
                                gyro_noise_model_->whitenError(raw_error_gyro);
                            const double sqrt_w_gyro = sqrt(gyro_loss_func_->weight(white_error_gyro.norm()));
                            const Eigen::Matrix<double, 3, 1> error_gyro = sqrt_w_gyro * white_error_gyro;

                            // Get Jacobians
                            Eigen::Matrix<double, 3, 36> G = Eigen::Matrix<double, 3, 36>::Zero();
                            G.block<3, 24>(0, 0) = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() *
                                                jac_vel_ * interp_jac_vel;
                            G.block<3, 12>(0, 24) = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() *
                                                    jac_bias_ * interp_jac_bias;

                            // Accumulate contributions
                            A += G.transpose() * G;
                            b += (-1) * G.transpose() * error_gyro;
                        } catch (const std::exception& e) {
                            ++exception_count;
                            std::cerr << "[GyroSuperCostTerm::buildGaussNewtonTerms] STEAM exception at index " << i
                                    << ", timestamp " << imu_data_vec_[i].timestamp << ": " << e.what() << std::endl;
                        } catch (...) {
                            ++exception_count;
                            std::cerr << "[GyroSuperCostTerm::buildGaussNewtonTerms] STEAM exception at index " << i
                                    << ", timestamp " << imu_data_vec_[i].timestamp << ": (unknown)" << std::endl;
                        }
                    }
                }
            );
        }

        // Combine thread-local A and b
        Matrix36x36 A = local_A.combine([](const Matrix36x36& a, const Matrix36x36& b) { return a + b; });
        Vector36 b = local_b.combine([](const Vector36& a, const Vector36& b) { return a + b; });

        // Determine active variables and extract keys
        std::vector<bool> active;
        active.push_back(knot1_->pose()->active());
        active.push_back(knot1_->velocity()->active());
        active.push_back(knot2_->pose()->active());
        active.push_back(knot2_->velocity()->active());
        active.push_back(bias1_->active());
        active.push_back(bias2_->active());

        std::vector<StateKey> keys;

        if (active[0]) {
            const auto T1node = std::static_pointer_cast<Node<PoseType>>(T1_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot1_->pose()->backward(lhs, T1node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
            keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }
        if (active[1]) {
            const auto w1node = std::static_pointer_cast<Node<VelType>>(w1_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot1_->velocity()->backward(lhs, w1node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
            keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }
        if (active[2]) {
            const auto T2node = std::static_pointer_cast<Node<PoseType>>(T2_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot2_->pose()->backward(lhs, T2node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
            keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }
        if (active[3]) {
            const auto w2node = std::static_pointer_cast<Node<VelType>>(w2_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot2_->velocity()->backward(lhs, w2node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
            keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }
        if (active[4]) {
            const auto b1node = std::static_pointer_cast<Node<BiasType>>(b1_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            bias1_->backward(lhs, b1node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
            keys.push_back(it->first);
            }

        } else {
            keys.push_back(-1);
        }
        if (active[5]) {
            const auto b2node = std::static_pointer_cast<Node<BiasType>>(b2_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            bias2_->backward(lhs, b2node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
            keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }

        // Update global Hessian and gradient for active variables
        static tbb::spin_mutex hessian_mutex;
        for (size_t i = 0; i < 6; ++i) {
            if (!active[i]) continue;
            const auto& key1 = keys[i];
            unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

            // Update gradient
            Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);
            static tbb::spin_mutex grad_mutex;
            tbb::spin_mutex::scoped_lock grad_lock(grad_mutex);
            gradient_vector->mapAt(blkIdx1) += newGradTerm;

            // Update Hessian (upper triangle)
            for (size_t j = i; j < 6; ++j) {
                if (!active[j]) continue;
                const auto& key2 = keys[j];
                unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

                unsigned int row, col;
                const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
                    if (blkIdx1 <= blkIdx2) {
                        row = blkIdx1;
                        col = blkIdx2;
                        return A.block<6, 6>(i * 6, j * 6);
                    } else {
                        row = blkIdx2;
                        col = blkIdx1;
                        return A.block<6, 6>(j * 6, i * 6);
                    }
                }();

                // Update Hessian with mutex protection
                tbb::spin_mutex::scoped_lock hessian_lock(hessian_mutex);
                BlockSparseMatrix::BlockRowEntry& entry = approximate_hessian->rowEntryAt(row, col, true);
                entry.data += newHessianTerm;
            }
        }

        // Log exceptions
        if (exception_count > 0) {
            std::cerr << "[GyroSuperCostTerm::buildGaussNewtonTerms] Warning: " << exception_count
                    << " exceptions occurred!" << std::endl;
        }
    }



}  // namespace finalicp