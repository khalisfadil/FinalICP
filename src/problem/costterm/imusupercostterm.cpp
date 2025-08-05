#include <problem/costterm/imusupercostterm.hpp> 

#include <iostream>

namespace finalicp {

    // ##########################################
    // MakeShared
    // ##########################################

    IMUSuperCostTerm::Ptr IMUSuperCostTerm::MakeShared(const Interface::ConstPtr &interface, const Time time1, const Time time2,
                                                        const Evaluable<BiasType>::ConstPtr &bias1,
                                                        const Evaluable<BiasType>::ConstPtr &bias2,
                                                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
                                                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
                                                        const Options &options) {
        return std::make_shared<IMUSuperCostTerm>(interface, time1, time2, bias1, bias2, transform_i_to_m_1, transform_i_to_m_2, options);
    }

    // ##########################################
    // cost
    // ##########################################

    double IMUSuperCostTerm::cost() const {

        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto dw1_ = knot1_->acceleration()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();
        const auto dw2_ = knot2_->acceleration()->forward();
        const auto b1_ = bias1_->forward();
        const auto b2_ = bias2_->forward();
        const auto T_mi_1_ = transform_i_to_m_1_->forward();
        const auto T_mi_2_ = transform_i_to_m_2_->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto dw1 = dw1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();
        const auto dw2 = dw2_->value();
        const auto b1 = b1_->value();
        const auto b2 = b2_->value();
        const auto T_mi_1 = T_mi_1_->value();
        const auto T_mi_2 = T_mi_2_->value();

        // Compute relative pose and velocity transformation
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto J_21_inv_w2 = J_21_inv * w2;
        const auto J_21_inv_curl_dw2 = (-0.5 * math::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

        // Sequential processing for small imu_data_vec_ to avoid parallel overhead
        double cost = 0;

#ifdef DEBUG
        std::cout << "[IMUSuperCostTerm DEBUG | cost] Calculating cost for " << imu_data_vec_.size() << " IMU measurements..." << std::endl;
#endif

        for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
            try {
                const double &ts = imu_data_vec_[i].timestamp;
                const IMUData &imu_data = imu_data_vec_[i];

                // Pose, velocity, acceleration interpolation
                const auto &omega = interp_mats_.at(ts).first;
                const auto &lambda = interp_mats_.at(ts).second;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + lambda(0, 2) * dw1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2 + omega(0, 2) * J_21_inv_curl_dw2;
                const Eigen::Matrix<double, 6, 1> xi_j1 = lambda(1, 1) * w1 + lambda(1, 2) * dw1 + omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2 + omega(1, 2) * J_21_inv_curl_dw2;
                const Eigen::Matrix<double, 6, 1> xi_k1 = lambda(2, 1) * w1 + lambda(2, 2) * dw1 + omega(2, 0) * xi_21 + omega(2, 1) * J_21_inv_w2 + omega(2, 2) * J_21_inv_curl_dw2;

                // Interpolated pose
                const math::se3::Transformation T_i1(xi_i1);
                const math::se3::Transformation T_i0 = T_i1 * T1;
                // Interpolated velocity
                const Eigen::Matrix<double, 6, 1> w_i = math::se3::vec2jac(xi_i1) * xi_j1;
                // Interpolated acceleration
                const Eigen::Matrix<double, 6, 1> dw_i = math::se3::vec2jac(xi_i1) * (xi_k1 + 0.5 * math::se3::curlyhat(xi_j1) * w_i);

                // Interpolated bias
                Eigen::Matrix<double, 6, 1> bias_i = Eigen::Matrix<double, 6, 1>::Zero();
                const double tau = ts - knot1_->time().seconds();
                const double T = knot2_->time().seconds() - knot1_->time().seconds();
                const double ratio = tau / T;
                const double omega_ = ratio;
                const double lambda_ = 1 - ratio;
                bias_i = lambda_ * b1 + omega_ * b2;

                // Interpolated T_mi
                math::se3::Transformation transform_i_to_m = T_mi_1;
                if (transform_i_to_m_1_->active() || transform_i_to_m_2_->active()) {
                    const double alpha_ = (ts - knot1_->time().seconds()) / (knot2_->time().seconds() - knot1_->time().seconds());
                    const Eigen::Matrix<double, 6, 1> xi_i1_ = alpha_ * (T_mi_2 / T_mi_1).vec();
                    transform_i_to_m = math::se3::Transformation(xi_i1_) * T_mi_1;
                }

                const Eigen::Matrix3d &C_vm = T_i0.matrix().block<3, 3>(0, 0);
                const Eigen::Matrix3d &C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);

                // Compute accelerometer error
                Eigen::Matrix<double, 3, 1> raw_error_acc = Eigen::Matrix<double, 3, 1>::Zero();
                if (options_.se2) {
                    raw_error_acc.block<2, 1>(0, 0) = imu_data.lin_acc.block<2, 1>(0, 0) + dw_i.block<2, 1>(0, 0) - bias_i.block<2, 1>(0, 0);
                } else {
                    raw_error_acc = imu_data.lin_acc + dw_i.block<3, 1>(0, 0) + C_vm * C_mi * options_.gravity - bias_i.block<3, 1>(0, 0);
                }

                // Compute gyroscope error
                Eigen::Matrix<double, 3, 1> raw_error_gyro = Eigen::Matrix<double, 3, 1>::Zero();
                if (options_.se2) {
                    raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                } else {
                    raw_error_gyro = imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                }

                // Evaluate cost
                double cost_i = 0.0;
                if (options_.use_accel) {
                    double acc_cost = acc_loss_func_->cost(acc_noise_model_->getWhitenedErrorNorm(raw_error_acc));
                    if (!std::isnan(acc_cost)) {cost_i += acc_cost; }
                }
                double gyro_cost = gyro_loss_func_->cost(gyro_noise_model_->getWhitenedErrorNorm(raw_error_gyro));
                if (!std::isnan(gyro_cost)) {cost_i += gyro_cost; }

                if (!std::isnan(cost_i)) {cost += cost_i; }

#ifdef DEBUG
                if (i == 0) {
                    std::cout << "[IMUSuperCostTerm DEBUG | cost] First IMU data point (t=" << std::fixed << std::setprecision(4) << ts << "):" << std::endl;
                    if (!dw_i.allFinite() || !w_i.allFinite()) {
                        std::cerr << "[IMUSuperCostTerm DEBUG | cost] CRITICAL: Interpolated state (vel/accel) is non-finite!" << std::endl;
                    } else {
                        std::cout << "[IMUSuperCostTerm DEBUG | cost] Interp Accel norm: " << dw_i.norm() << ", Interp Vel norm: " << w_i.norm() << std::endl;
                        std::cout << "[IMUSuperCostTerm DEBUG | cost] Interp Bias norm:  " << bias_i.norm() << std::endl;
                    }

                    if (!raw_error_acc.allFinite() || !raw_error_gyro.allFinite()) {
                        std::cerr << "[IMUSuperCostTerm DEBUG | cost] CRITICAL: Raw IMU error is non-finite!" << std::endl;
                    } else {
                        std::cout << "[IMUSuperCostTerm DEBUG | cost] Raw Accel Error norm: " << raw_error_acc.norm() << std::endl;
                        std::cout << "[IMUSuperCostTerm DEBUG | cost] Raw Gyro Error norm:  " << raw_error_gyro.norm() << std::endl;
                    }
                }
#endif

            } catch (const std::exception& e) {
                std::cerr << "[IMUSuperCostTerm::cost] exception at timestamp " << imu_data_vec_[i].timestamp << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[IMUSuperCostTerm::cost] exception at timestamp " << imu_data_vec_[i].timestamp << ": (unknown)" << std::endl;
            }
        }

#ifdef DEBUG
        std::cout << "[IMUSuperCostTerm DEBUG | cost] Total IMU cost contribution: " << cost << std::endl;
#endif

        return cost;
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    void IMUSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
        knot1_->pose()->getRelatedVarKeys(keys);
        knot1_->velocity()->getRelatedVarKeys(keys);
        knot1_->acceleration()->getRelatedVarKeys(keys);
        knot2_->pose()->getRelatedVarKeys(keys);
        knot2_->velocity()->getRelatedVarKeys(keys);
        knot2_->acceleration()->getRelatedVarKeys(keys);
        bias1_->getRelatedVarKeys(keys);
        bias2_->getRelatedVarKeys(keys);
        transform_i_to_m_1_->getRelatedVarKeys(keys);
        transform_i_to_m_2_->getRelatedVarKeys(keys);
    }

    // ##########################################
    // init
    // ##########################################

    void IMUSuperCostTerm::init() { initialize_interp_matrices_(); }

    // ##########################################
    // initialize_interp_matrices_
    // ##########################################

    void IMUSuperCostTerm::initialize_interp_matrices_() {
#ifdef DEBUG
        std::cout << "[IMUSuperCostTerm DEBUG | initialize_interp_matrices_] Initializing interpolation matrices for " << imu_data_vec_.size() << " timestamps." << std::endl;
#endif
        const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
        // #pragma omp parallel for num_threads(options_.num_threads)
        for (const IMUData &imu_data : imu_data_vec_) {
            const double &time = imu_data.timestamp;
            if (interp_mats_.find(time) == interp_mats_.end()) {
            // Get Lambda, Omega for this time
            const double tau = time - time1_.seconds();
            const double kappa = knot2_->time().seconds() - time;
            const Matrix18d Q_tau = interface_->getQPublic(tau, ones);
            const Matrix18d Tran_kappa = interface_->getTranPublic(kappa);
            const Matrix18d Tran_tau = interface_->getTranPublic(tau);
            const Matrix18d omega18 = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
            const Matrix18d lambda18 = (Tran_tau - omega18 * Tran_T_);
            Eigen::Matrix3d omega = Eigen::Matrix3d::Zero();
            Eigen::Matrix3d lambda = Eigen::Matrix3d::Zero();
            omega(0, 0) = omega18(0, 0);
            omega(0, 1) = omega18(0, 6);
            omega(0, 2) = omega18(0, 12);
            omega(1, 0) = omega18(6, 0);
            omega(1, 1) = omega18(6, 6);
            omega(1, 2) = omega18(6, 12);
            omega(2, 0) = omega18(12, 0);
            omega(2, 1) = omega18(12, 6);
            omega(2, 2) = omega18(12, 12);
            lambda(0, 0) = lambda18(0, 0);
            lambda(0, 1) = lambda18(0, 6);
            lambda(0, 2) = lambda18(0, 12);
            lambda(1, 0) = lambda18(6, 0);
            lambda(1, 1) = lambda18(6, 6);
            lambda(1, 2) = lambda18(6, 12);
            lambda(2, 0) = lambda18(12, 0);
            lambda(2, 1) = lambda18(12, 6);
            lambda(2, 2) = lambda18(12, 12);
            const auto omega_lambda = std::make_pair(omega, lambda);
#ifdef DEBUG
            // --- [IMPROVEMENT] Sanity-check the computed matrices ---
            if (!omega.allFinite() || !lambda.allFinite()) {
                    std::cerr << "[IMUSuperCostTerm DEBUG | initialize_interp_matrices_] CRITICAL: Computed interpolation matrices for time " << time << " are non-finite!" << std::endl;
            }
#endif
            interp_mats_.emplace(time, omega_lambda);
            }
        }
    }

    // ##########################################
    // buildGaussNewtonTerms
    // ##########################################

    void IMUSuperCostTerm::buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const {
#ifdef DEBUG
        std::cout << "[IMUSuperCostTerm DEBUG | buildGaussNewtonTerms] Building Gauss-Newton terms..." << std::endl;
#endif
        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto dw1_ = knot1_->acceleration()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();
        const auto dw2_ = knot2_->acceleration()->forward();
        const auto b1_ = bias1_->forward();
        const auto b2_ = bias2_->forward();
        const auto T_mi_1_ = transform_i_to_m_1_->forward();
        const auto T_mi_2_ = transform_i_to_m_2_->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto dw1 = dw1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();
        const auto dw2 = dw2_->value();
        const auto b1 = b1_->value();
        const auto b2 = b2_->value();
        const auto T_mi_1 = T_mi_1_->value();
        const auto T_mi_2 = T_mi_2_->value();

        // Compute relative pose and velocity transformation
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const auto Ad_T_21 = math::se3::tranAd(T_21.matrix());
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto J_21_inv_w2 = J_21_inv * w2;
        const auto J_21_inv_curl_dw2 = (-0.5 * math::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

        // Thread-local accumulators for A and b
        using Matrix60x60 = Eigen::Matrix<double, 60, 60>;
        using Vector60 = Eigen::Matrix<double, 60, 1>;

        // Process IMU data: sequential for small sizes, parallel for large
        Matrix60x60 A = Matrix60x60::Zero();
        Vector60 b = Vector60::Zero();
        for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
            try {
                const double &ts = imu_data_vec_[i].timestamp;
                const IMUData &imu_data = imu_data_vec_[i];

                // Interpolation
                const auto &omega = interp_mats_.at(ts).first;
                const auto &lambda = interp_mats_.at(ts).second;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + lambda(0, 2) * dw1 + omega(0, 0) * xi_21 + omega(0, 6) * J_21_inv_w2 + omega(0, 2) * J_21_inv_curl_dw2;
                const Eigen::Matrix<double, 6, 1> xi_j1 = lambda(1, 1) * w1 + lambda(1, 2) * dw1 + omega(1, 0) * xi_21 + omega(1, 1) * J_21_inv_w2 + omega(1, 2) * J_21_inv_curl_dw2;
                const Eigen::Matrix<double, 6, 1> xi_k1 = lambda(2, 1) * w1 + lambda(2, 2) * dw1 + omega(2, 0) * xi_21 + omega(2, 1) * J_21_inv_w2 + omega(2, 2) * J_21_inv_curl_dw2;

                // Interpolated pose, velocity, acceleration
                const math::se3::Transformation T_i1(xi_i1);
                const math::se3::Transformation T_i0 = T_i1 * T1;
                const Eigen::Matrix<double, 6, 1> w_i = math::se3::vec2jac(xi_i1) * xi_j1;
                const Eigen::Matrix<double, 6, 1> dw_i = math::se3::vec2jac(xi_i1) * (xi_k1 + 0.5 * math::se3::curlyhat(xi_j1) * w_i);

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

                // Interpolated T_mi
                math::se3::Transformation transform_i_to_m = T_mi_1;
                Eigen::Matrix<double, 6, 12> interp_jac_T_m_i = Eigen::Matrix<double, 6, 12>::Zero();
                if (transform_i_to_m_1_->active() || transform_i_to_m_2_->active()) {
                    const double alpha_ = (ts - knot1_->time().seconds()) / (knot2_->time().seconds() - knot1_->time().seconds());
                    const Eigen::Matrix<double, 6, 1> xi_i1_ = alpha_ * (T_mi_2 / T_mi_1).vec();
                    transform_i_to_m = math::se3::Transformation(xi_i1_) * T_mi_1;
                    std::vector<double> faulhaber_coeffs_;
                    faulhaber_coeffs_.push_back(alpha_);
                    faulhaber_coeffs_.push_back(alpha_ * (alpha_ - 1) / 2);
                    faulhaber_coeffs_.push_back(alpha_ * (alpha_ - 1) * (2 * alpha_ - 1) / 12);
                    faulhaber_coeffs_.push_back(alpha_ * alpha_ * (alpha_ - 1) * (alpha_ - 1) / 24);
                    const Eigen::Matrix<double, 6, 6> xi_21_curlyhat = math::se3::curlyhat((T_mi_2 / T_mi_1).vec());
                    Eigen::Matrix<double, 6, 6> A_tmp = Eigen::Matrix<double, 6, 6>::Zero();
                    Eigen::Matrix<double, 6, 6> xictmp = Eigen::Matrix<double, 6, 6>::Identity();
                    for (unsigned int j = 0; j < faulhaber_coeffs_.size(); j++) {
                        if (j > 0) xictmp = xi_21_curlyhat * xictmp;
                        A_tmp += faulhaber_coeffs_[j] * xictmp;
                    }
                    interp_jac_T_m_i.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity() - A_tmp;
                    interp_jac_T_m_i.block<6, 6>(0, 6) = A_tmp;
                }

                // Jacobians
                Eigen::Matrix<double, 6, 36> interp_jac_pose = Eigen::Matrix<double, 6, 36>::Zero();
                Eigen::Matrix<double, 6, 36> interp_jac_vel = Eigen::Matrix<double, 6, 36>::Zero();
                Eigen::Matrix<double, 6, 36> interp_jac_acc = Eigen::Matrix<double, 6, 36>::Zero();
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 6> xi_j1_ch = -0.5 * math::se3::curlyhat(xi_j1);
                const auto J_prep_2 = J_i1 * (-0.5 * math::se3::curlyhat(w_i) + 0.5 * math::se3::curlyhat(xi_j1) * J_i1);
                const auto J_prep_3 = -0.25 * J_i1 * math::se3::curlyhat(xi_j1) * math::se3::curlyhat(xi_j1) - 0.5 * math::se3::curlyhat(xi_k1 + 0.5 * math::se3::curlyhat(xi_j1) * w_i);

                // Pose interpolation Jacobian
                Eigen::Matrix<double, 6, 6> w = J_i1 * (omega(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega(0, 1) * 0.5 * math::se3::curlyhat(w2) + omega(0, 2) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega(0, 2) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv;
                interp_jac_pose.block<6, 6>(0, 0) = -w * Ad_T_21 + T_i1.adjoint();
                interp_jac_pose.block<6, 6>(0, 6) = lambda(0, 1) * J_i1;
                interp_jac_pose.block<6, 6>(0, 12) = lambda(0, 2) * J_i1;
                interp_jac_pose.block<6, 6>(0, 18) = w;
                interp_jac_pose.block<6, 6>(0, 24) = omega(0, 1) * J_i1 * J_21_inv + omega(0, 2) * -0.5 * J_i1 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv);
                interp_jac_pose.block<6, 6>(0, 30) = omega(0, 2) * J_i1 * J_21_inv;

                // Velocity interpolation Jacobian
                w = J_i1 * (omega(1, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega(1, 1) * 0.5 * math::se3::curlyhat(w2) + omega(1, 2) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega(1, 2) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv + xi_j1_ch * (omega(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega(0, 1) * 0.5 * math::se3::curlyhat(w2) + omega(0, 2) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega(0, 2) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv;
                interp_jac_vel.block<6, 6>(0, 0) = -w * Ad_T_21;
                interp_jac_vel.block<6, 6>(0, 6) = J_i1 * lambda(1, 1) + xi_j1_ch * lambda(0, 1);
                interp_jac_vel.block<6, 6>(0, 12) = J_i1 * lambda(1, 2) + xi_j1_ch * lambda(0, 2);
                interp_jac_vel.block<6, 6>(0, 18) = w;
                interp_jac_vel.block<6, 6>(0, 24) = J_i1 * (omega(1, 1) * J_21_inv + omega(1, 2) * -0.5 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv)) + xi_j1_ch * (omega(0, 1) * J_21_inv + omega(0, 2) * -0.5 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv));
                interp_jac_vel.block<6, 6>(0, 30) = J_i1 * (omega(1, 2) * J_21_inv) + xi_j1_ch * (omega(0, 2) * J_21_inv);

                // Acceleration interpolation Jacobian
                w = J_i1 * (omega(2, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega(2, 1) * 0.5 * math::se3::curlyhat(w2) + omega(2, 2) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega(2, 2) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv + J_prep_2 * (omega(1, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega(1, 1) * 0.5 * math::se3::curlyhat(w2) + omega(1, 2) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega(1, 2) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv + J_prep_3 * (omega(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega(0, 1) * 0.5 * math::se3::curlyhat(w2) + omega(0, 2) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega(0, 2) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv;
                interp_jac_acc.block<6, 6>(0, 0) = -w * Ad_T_21;
                interp_jac_acc.block<6, 6>(0, 6) = J_i1 * lambda(2, 1) + J_prep_2 * lambda(1, 1) + J_prep_3 * lambda(0, 1);
                interp_jac_acc.block<6, 6>(0, 12) = J_i1 * lambda(2, 2) + J_prep_2 * lambda(1, 2) + J_prep_3 * lambda(0, 2);
                interp_jac_acc.block<6, 6>(0, 18) = w;
                interp_jac_acc.block<6, 6>(0, 24) = J_i1 * (omega(2, 1) * J_21_inv + omega(2, 2) * -0.5 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv)) + J_prep_2 * (omega(1, 1) * J_21_inv + omega(1, 2) * -0.5 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv)) + J_prep_3 * (omega(0, 1) * J_21_inv + omega(0, 2) * -0.5 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv));
                interp_jac_acc.block<6, 6>(0, 30) = J_i1 * (omega(2, 2) * J_21_inv) + J_prep_2 * (omega(1, 2) * J_21_inv) + J_prep_3 * (omega(0, 2) * J_21_inv);

                // Measurement Jacobians
                const Eigen::Matrix3d &C_vm = T_i0.matrix().block<3, 3>(0, 0);
                const Eigen::Matrix3d &C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);
                Eigen::Matrix<double, 3, 1> raw_error_acc = Eigen::Matrix<double, 3, 1>::Zero();
                if (options_.se2) {
                    raw_error_acc.block<2, 1>(0, 0) = imu_data.lin_acc.block<2, 1>(0, 0) + dw_i.block<2, 1>(0, 0) - bias_i.block<2, 1>(0, 0);
                } else {
                    raw_error_acc = imu_data.lin_acc + dw_i.block<3, 1>(0, 0) + C_vm * C_mi * options_.gravity - bias_i.block<3, 1>(0, 0);
                }
                const Eigen::Matrix<double, 3, 1> white_error_acc = acc_noise_model_->whitenError(raw_error_acc);
                const double sqrt_w_acc = sqrt(acc_loss_func_->weight(white_error_acc.norm()));
                const Eigen::Matrix<double, 3, 1> error_acc = sqrt_w_acc * white_error_acc;

                Eigen::Matrix<double, 3, 24> Gmeas = Eigen::Matrix<double, 3, 24>::Zero();
                Gmeas.block<3, 6>(0, 6) = jac_accel_;
                Gmeas.block<3, 6>(0, 12) = jac_bias_accel_;
                if (options_.se2) {
                    Gmeas.block<1, 24>(2, 0).setZero();
                }
                if (!options_.se2 && options_.use_accel) {
                    Gmeas.block<3, 3>(0, 3) = -1 * math::so3::hat(C_vm * C_mi * options_.gravity);
                    Gmeas.block<3, 3>(0, 21) = -1 * C_vm * math::so3::hat(C_mi * options_.gravity);
                }
                Gmeas = sqrt_w_acc * acc_noise_model_->getSqrtInformation() * Gmeas;

                Eigen::Matrix<double, 3, 1> raw_error_gyro = Eigen::Matrix<double, 3, 1>::Zero();
                if (options_.se2) {
                    raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
                } else {
                    raw_error_gyro = imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
                }
                const Eigen::Matrix<double, 3, 1> white_error_gyro = gyro_noise_model_->whitenError(raw_error_gyro);
                const double sqrt_w_gyro = sqrt(gyro_loss_func_->weight(white_error_gyro.norm()));
                const Eigen::Matrix<double, 3, 1> error_gyro = sqrt_w_gyro * white_error_gyro;

                // Combine Jacobians
                Eigen::Matrix<double, 6, 60> G = Eigen::Matrix<double, 6, 60>::Zero();
                G.block<3, 36>(0, 0) = Gmeas.block<3, 6>(0, 0) * interp_jac_pose + Gmeas.block<3, 6>(0, 6) * interp_jac_acc;
                G.block<3, 12>(0, 36) = Gmeas.block<3, 6>(0, 12) * interp_jac_bias;
                G.block<3, 12>(0, 48) = Gmeas.block<3, 6>(0, 18) * interp_jac_T_m_i;
                G.block<3, 36>(3, 0) = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() * jac_vel_ * interp_jac_vel;
                G.block<3, 12>(3, 36) = sqrt_w_gyro * gyro_noise_model_->getSqrtInformation() * jac_bias_gyro_ * interp_jac_bias;

                Eigen::Matrix<double, 6, 1> error = Eigen::Matrix<double, 6, 1>::Zero();
                error.block<3, 1>(0, 0) = error_acc;
                error.block<3, 1>(3, 0) = error_gyro;

                // Accumulate contributions
                A += G.transpose() * G;
                b += (-1) * G.transpose() * error;

#ifdef DEBUG
                // --- [IMPROVEMENT] Check Jacobians for the first measurement ---
                if (i == 0) {
                    if (!interp_jac_pose.allFinite() || !interp_jac_vel.allFinite() || !interp_jac_acc.allFinite()) {
                        std::cerr << "[IMUSuperCostTerm DEBUG | buildGaussNewtonTerms] CRITICAL: Intermediate state Jacobians are non-finite!" << std::endl;
                    }
                    if (!G.allFinite() || !error.allFinite()) {
                        std::cerr << "[IMUSuperCostTerm DEBUG | buildGaussNewtonTerms] CRITICAL: Final Jacobian (G) or error vector is non-finite!" << std::endl;
                    } else {
                        std::cout << "[IMUSuperCostTerm DEBUG | buildGaussNewtonTerms] First IMU point Jacobian norm (G): " << G.norm() << std::endl;
                    }
                }
#endif
            } catch (const std::exception& e) {
                std::cerr << "[IMUSuperCostTerm::buildGaussNewtonTerms] exception at timestamp " << imu_data_vec_[i].timestamp << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[IMUSuperCostTerm::buildGaussNewtonTerms] exception at timestamp " << imu_data_vec_[i].timestamp << ": (unknown)" << std::endl;
            }
        }

#ifdef DEBUG
        // --- [IMPROVEMENT] Check the accumulated local system before scattering ---
        if (!A.allFinite() || !b.allFinite()) {
            std::cerr << "[IMUSuperCostTerm DEBUG] CRITICAL: Accumulated local Hessian (A) or Gradient (b) is non-finite!" << std::endl;
        } else {
             std::cout << "[IMUSuperCostTerm DEBUG | buildGaussNewtonTerms] Accumulated local Hessian norm: " << A.norm() << ", Gradient norm: " << b.norm() << std::endl;
        }
#endif

        // Determine active variables and extract keys
        std::vector<bool> active;
        active.push_back(knot1_->pose()->active());
        active.push_back(knot1_->velocity()->active());
        active.push_back(knot1_->acceleration()->active());
        active.push_back(knot2_->pose()->active());
        active.push_back(knot2_->velocity()->active());
        active.push_back(knot2_->acceleration()->active());
        active.push_back(bias1_->active());
        active.push_back(bias2_->active());
        active.push_back(transform_i_to_m_1_->active());
        active.push_back(transform_i_to_m_2_->active());

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
            const auto dw1node = std::static_pointer_cast<Node<AccType>>(dw1_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot1_->acceleration()->backward(lhs, dw1node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }
        if (active[3]) {
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
        if (active[4]) {
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
        if (active[5]) {
            const auto dw2node = std::static_pointer_cast<Node<AccType>>(dw2_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot2_->acceleration()->backward(lhs, dw2node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }
        if (active[6]) {
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
        if (active[7]) {
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
        if (active[8]) {
            const auto T_mi_1_node = std::static_pointer_cast<Node<PoseType>>(T_mi_1_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            transform_i_to_m_1_->backward(lhs, T_mi_1_node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }
        if (active[9]) {
            const auto T_mi_2_node = std::static_pointer_cast<Node<PoseType>>(T_mi_2_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            transform_i_to_m_2_->backward(lhs, T_mi_2_node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        } else {
            keys.push_back(-1);
        }

        // Update global Hessian and gradient for active variables
        for (size_t i = 0; i < 10; ++i) {
            try {
                if (!active[i]) continue;

                const auto& key1 = keys[i];
                unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

                // Update gradient
                Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);


                gradient_vector->mapAt(blkIdx1) += newGradTerm;

                // Update Hessian (upper triangle)
                for (size_t j = i; j < 10; ++j) {

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
                    
                    BlockSparseMatrix::BlockRowEntry& entry = approximate_hessian->rowEntryAt(row, col, true);

                    // omp_set_lock(&entry.lock);
                    entry.data += newHessianTerm;
                    // omp_unset_lock(&entry.lock);

// #ifdef DEBUG
//                     // --- [IMPROVEMENT] Add logging to the scatter process ---
//                     std::cout << "    - Scattering contribution from local block (" << i << "," << j << ") to global block (" << row << "," << col << ")" << std::endl;
// #endif
                }
            } catch (const std::exception& e) {
                std::cerr << "[IMUSuperCostTerm::buildGaussNewtonTerms] exception at index " << i << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[IMUSuperCostTerm::buildGaussNewtonTerms] exception at index " << i << ": (unknown)" << std::endl;
            }
        }
    }
}  // namespace finalicp