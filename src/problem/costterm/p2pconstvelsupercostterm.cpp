#include <iostream>

#include <problem/costterm/p2pconstvelsupercostterm.hpp> 

namespace finalicp {

    P2PCVSuperCostTerm::Ptr P2PCVSuperCostTerm::MakeShared(const Interface::ConstPtr &interface, const Time time1, const Time time2, const Options &options) {
        return std::make_shared<P2PCVSuperCostTerm>(interface, time1, time2, options);
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    double P2PCVSuperCostTerm::cost() const {

        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();

        // Compute relative pose and velocity transformation
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto J_21_inv_w2 = J_21_inv * w2;

        const double rinv = 1.0 / options_.r_p2p;
        const double sqrt_rinv = sqrt(rinv);

        // Sequential processing for small meas_times_ to avoid parallel overhead
#ifdef DEBUG
        std::cout << "[P2PCVSuperCostTerm DEBUG | cost] Calculating cost for " << p2p_matches_.size() << " matches across " << meas_times_.size() << " timestamps..." << std::endl;
#endif  
        double cost = 0;
        for (unsigned int i = 0; i < meas_times_.size(); ++i) {
            try {
                const double &ts = meas_times_[i];
                const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);

                // Pose interpolation
                const auto &omega = interp_mats_.at(ts).first;
                const auto &lambda = interp_mats_.at(ts).second;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                const math::se3::Transformation T_i1(xi_i1);
                const math::se3::Transformation T_i0 = T_i1 * T1;
                const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

                double cost_i = 0.0;
                for (const int &match_idx : bin_indices) {
                    const auto &p2p_match = p2p_matches_.at(match_idx);
                    const double raw_error = p2p_match.normal.transpose() * (p2p_match.reference - T_mr.block<3, 3>(0, 0) * p2p_match.query - T_mr.block<3, 1>(0, 3));
                    double match_cost = p2p_loss_func_->cost(sqrt_rinv * fabs(raw_error));
                    if (!std::isnan(match_cost)) {cost_i += match_cost; }
#ifdef DEBUG
                    // Log details for the very first point-to-plane match only to avoid spam
                    if (i == 0 && match_idx == bin_indices[0]) {
                        if (!std::isfinite(raw_error)) {
                             std::cerr << "[P2PCVSuperCostTerm DEBUG | cost] CRITICAL: Raw P2P error is non-finite!" << std::endl;
                        } else {
                            std::cout << "[P2PCVSuperCostTerm DEBUG | cost] First match (t=" << std::fixed << std::setprecision(4) << ts << "): raw_error = " << raw_error << std::endl;
                        }
                    }
#endif
                }

                if (!std::isnan(cost_i)) {cost += cost_i; }
            } catch (const std::exception& e) {
                std::cerr << "[P2PCVSuperCostTerm | cost] exception at timestamp " << meas_times_[i] << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[P2PCVSuperCostTerm | cost] exception at timestamp " << meas_times_[i] << ": (unknown)" << std::endl;
            }
        }
        return cost;
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    void P2PCVSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
        knot1_->pose()->getRelatedVarKeys(keys);
        knot1_->velocity()->getRelatedVarKeys(keys);
        knot2_->pose()->getRelatedVarKeys(keys);
        knot2_->velocity()->getRelatedVarKeys(keys);
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    void P2PCVSuperCostTerm::initP2PMatches() {
#ifdef DEBUG
        std::cout << "[P2PCVSuperCostTerm DEBUG | initP2PMatches] Initializing... p2p_matches_ " << p2p_matches_.size() << " matches by timestamp." << std::endl;
#endif
        p2p_match_bins_.clear();
        if (p2p_matches_.size() < static_cast<size_t>(sequential_threshold_)){
            // sequential method 
            for (int i = 0; i < static_cast<int>(p2p_matches_.size()); ++i) {
                const auto &p2p_match = p2p_matches_.at(i);
                const auto &timestamp = p2p_match.timestamp;
                if (p2p_match_bins_.find(timestamp) == p2p_match_bins_.end()) {
                    p2p_match_bins_[timestamp] = {i};
                } else {
                    p2p_match_bins_[timestamp].push_back(i);
                }
            }
        } else {
            // parallel method
            // --- Parallel Path for large match sets ---
            using LocalMatchBinType = std::map<double, std::vector<int>>;
            tbb::enumerable_thread_specific<LocalMatchBinType> thread_local_match_bins;
            tbb::parallel_for(tbb::blocked_range<int>(0, p2p_matches_.size()),
                [&](const tbb::blocked_range<int>& range) {
                    LocalMatchBinType& local_map = thread_local_match_bins.local();
                    for (int i = range.begin(); i != range.end(); ++i) {
                        const auto &p2p_match = p2p_matches_.at(i);
                        const auto &timestamp = p2p_match.timestamp;
                        if (local_map.find(timestamp) == local_map.end()) {
                            local_map[timestamp] = {i};
                        } else {
                            local_map[timestamp].push_back(i);
                        }
                    }
                });

            // Merge the thread-local maps into the main bin sequentially
            for (const auto& local_map : thread_local_match_bins) {
                for (const auto& pair : local_map) {
                    p2p_match_bins_[pair.first].insert(
                        p2p_match_bins_[pair.first].end(),
                        pair.second.begin(),
                        pair.second.end()
                    );
                }
            }
        }

        meas_times_.clear();
        meas_times_.reserve(p2p_match_bins_.size());
        for (auto it = p2p_match_bins_.begin(); it != p2p_match_bins_.end(); it++) {
            meas_times_.push_back(it->first);
        }
        initialize_interp_matrices_();
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    void P2PCVSuperCostTerm::initialize_interp_matrices_() {
#ifdef DEBUG
        std::cout << "[P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_] Initializing... meas_times_ " << meas_times_.size() << std::endl;
#endif
        interp_mats_.clear();
        const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
        if (meas_times_.size() < static_cast<size_t>(sequential_threshold_)){
            for (const double &time : meas_times_) {
                if (interp_mats_.find(time) == interp_mats_.end()) { // if time does not exits
                    const double tau = (Time(time) - time1_).seconds();

                    Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
                    Eigen::Matrix4d lambda = Eigen::Matrix4d::Zero();
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
        } else {
            // Thread-local storage for interpolation matrices
            using LocalInterpMapType = std::map<double, std::pair<Eigen::Matrix4d, Eigen::Matrix4d>>;
            tbb::enumerable_thread_specific<LocalInterpMapType> thread_local_interp_maps;

            // --- 1. Parallel Computation into Local Containers ---
            tbb::parallel_for(tbb::blocked_range<size_t>(0, meas_times_.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    // Get this thread's private map
                    LocalInterpMapType& local_map = thread_local_interp_maps.local();
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const double& time = meas_times_[i];
                        // Skip if already computed (optional, depending on requirements)
                        if (local_map.find(time) != local_map.end()) continue;              
                        const double tau = (Time(time) - time1_).seconds();
                        const double kappa = knot2_->time().seconds() - time;
                        // Compute matrices
                        const Matrix12d Q_tau = traj::const_vel::getQ(tau, ones);
                        const Matrix12d Tran_kappa = traj::const_vel::getTran(kappa);
                        const Matrix12d Tran_tau = traj::const_vel::getTran(tau);
                        const Matrix12d omega12 = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
                        const Matrix12d lambda12 = (Tran_tau - omega12 * Tran_T_);
                        // Populate 4x4 matrices (only 2x2 submatrix used)
                        Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
                        Eigen::Matrix4d lambda = Eigen::Matrix4d::Zero();
                        omega(0, 0) = omega12(0, 0); omega(0, 1) = omega12(0, 6);
                        omega(1, 0) = omega12(6, 0); omega(1, 1) = omega12(6, 6);
                        lambda(0, 0) = lambda12(0, 0); lambda(0, 1) = lambda12(0, 6);
                        lambda(1, 0) = lambda12(6, 0); lambda(1, 1) = lambda12(6, 6);
                        // Anomaly checking
#ifdef DEBUG
                        if (!omega.allFinite() || !lambda.allFinite()) {
                            std::cerr << "[P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_] CRITICAL: Computed interpolation matrices for time " << time << " are non-finite!" << std::endl;
                        }
#endif
                        // Store in thread-local map
                        local_map.emplace(time, std::make_pair(omega, lambda));
                    }
                });
            // --- 2. Sequential Merge Step ---
            for (const auto& local_map : thread_local_interp_maps) {
                interp_mats_.insert(local_map.begin(), local_map.end());
            }
        }
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    void P2PCVSuperCostTerm::buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const {

        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();

        // Compute relative pose and velocity transformation
        const double rinv = 1.0 / options_.r_p2p;
        const double sqrt_rinv = sqrt(rinv);
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const auto Ad_T_21 = math::se3::tranAd(T_21.matrix());
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto w2_j_21_inv = 0.5 * math::se3::curlyhat(w2) * J_21_inv;
        const auto J_21_inv_w2 = J_21_inv * w2;

        // Thread-local accumulators for A and b
        using Matrix24x24 = Eigen::Matrix<double, 24, 24>;
        using Vector24 = Eigen::Matrix<double, 24, 1>;

        // Process measurement times: sequential for small sizes, parallel for large
        Matrix24x24 A = Matrix24x24::Zero();
        Vector24 b = Vector24::Zero();
        for (int i = 0; i < (int)meas_times_.size(); ++i) {
            try {
                const double &ts = meas_times_[i];
                const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);

                // Pose interpolation
                const auto &omega = interp_mats_.at(ts).first;
                const auto &lambda = interp_mats_.at(ts).second;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                const math::se3::Transformation T_i1(xi_i1);
                const math::se3::Transformation T_i0 = T_i1 * T1;
                const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

                // Pose interpolation Jacobians
                Eigen::Matrix<double, 6, 24> interp_jac = Eigen::Matrix<double, 6, 24>::Zero();
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 6> w = J_i1 * (omega(0, 0) * J_21_inv + omega(0, 1) * w2_j_21_inv);
                interp_jac.block<6, 6>(0, 0) = -w * Ad_T_21 + T_i1.adjoint(); // T1
                interp_jac.block<6, 6>(0, 6) = lambda(0, 1) * J_i1; // w1
                interp_jac.block<6, 6>(0, 12) = w; // T2
                interp_jac.block<6, 6>(0, 18) = omega(0, 1) * J_i1 * J_21_inv; // w2

                // Measurement Jacobians
                Eigen::Matrix<double, 1, 6> Gmeas = Eigen::Matrix<double, 1, 6>::Zero();
                double error = 0.0;

                for (const int &match_idx : bin_indices) {
                    const auto &p2p_match = p2p_matches_.at(match_idx);
                    const double raw_error = p2p_match.normal.transpose() * (p2p_match.reference - T_mr.block<3, 3>(0, 0) * p2p_match.query - T_mr.block<3, 1>(0, 3));
                    const double sqrt_w = sqrt(p2p_loss_func_->weight(fabs(raw_error)));
                    error += sqrt_w * sqrt_rinv * raw_error;
                    Gmeas += sqrt_w * sqrt_rinv * p2p_match.normal.transpose() * (T_mr * math::se3::point2fs(p2p_match.query)).block<3, 6>(0, 0);
                }

                const Eigen::Matrix<double, 1, 24> G = Gmeas * interp_jac;
                A += G.transpose() * G;
                b -= G.transpose() * error;
#ifdef DEBUG
                if (i == 0) {
                    if (!interp_jac.allFinite()) {
                        std::cerr << "[P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms] CRITICAL: Pose interpolation Jacobian (interp_jac) is non-finite!" << std::endl;
                    }
                    if (!Gmeas.allFinite()) {
                        std::cerr << "[P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms] CRITICAL: Aggregated measurement Jacobian (Gmeas) is non-finite!" << std::endl;
                    } else {
                        std::cout << "[P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms] First timestamp bin: interp_jac norm: " << interp_jac.norm() << ", Gmeas norm: " << Gmeas.norm() << std::endl;
                    }
                }
#endif
            } catch (const std::exception& e) {
                std::cerr << "[P2PCVSuperCostTerm | buildGaussNewtonTerms] exception at timestamp " << meas_times_[i] << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[P2PCVSuperCostTerm | buildGaussNewtonTerms] exception at timestamp " << meas_times_[i] << ": (unknown)" << std::endl;
            }
        }
#ifdef DEBUG
        // --- [IMPROVEMENT] Check the accumulated local system before scattering ---
        if (!A.allFinite() || !b.allFinite()) {
            std::cerr << "[P2PCVSuperCostTerm DEBUG] CRITICAL: Accumulated local Hessian (A) or Gradient (b) is non-finite!" << std::endl;
        } else {
             std::cout << "[P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms] Accumulated local Hessian norm: " << A.norm() << ", Gradient norm: " << b.norm() << std::endl;
        }
#endif

        // Determine active variables and extract keys
        std::vector<bool> active;
        active.push_back(knot1_->pose()->active());
        active.push_back(knot1_->velocity()->active());
        active.push_back(knot2_->pose()->active());
        active.push_back(knot2_->velocity()->active());

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

        // Update global Hessian and gradient for active variables
        for (size_t i = 0; i < 4; ++i) {
            try {
                if (!active[i]) continue;
                const auto& key1 = keys[i];
                unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

                // Debug: Print key, Jacobian size, and block index
                // ################################
                // std::cout << "[DEBUG] i=" << i << ", key1=" << key1 << ", blkIdx1=" << blkIdx1 << std::endl;
                // ################################

                // Update gradient
                Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

                // Debug: Print gradient term size and norm
                // ################################
                // std::cout << "[DEBUG] Gradient term size: (" << newGradTerm.rows() << ", " << newGradTerm.cols() << "), norm: " << newGradTerm.norm() << std::endl;
                // ################################

                gradient_vector->mapAt(blkIdx1) += newGradTerm;

                // Update Hessian (upper triangle)
                for (size_t j = i; j < 4; ++j) {
                    if (!active[j]) continue;
                    const auto& key2 = keys[j];
                    unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

                    // Debug: Print inner loop key and block index
                    // ################################
                    // std::cout << "[DEBUG] j=" << j << ", key2=" << key2 << ", blkIdx2=" << blkIdx2 << std::endl;
                    // ################################

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

                    // Debug: Print Hessian term size and norm
                    // ################################
                    // std::cout << "[DEBUG] Hessian term (row=" << row << ", col=" << col << ") size: (" << newHessianTerm.rows() << ", " << newHessianTerm.cols() << "), norm: " << newHessianTerm.norm() << std::endl;
                    // ################################

                    // Update Hessian with mutex protection
                    BlockSparseMatrix::BlockRowEntry& entry = approximate_hessian->rowEntryAt(row, col, true);
                    // omp_set_lock(&entry.lock);
                    entry.data += newHessianTerm;
                    // omp_unset_lock(&entry.lock);
                }
            } catch (const std::exception& e) {
                std::cerr << "[P2PCVSuperCostTerm | buildGaussNewtonTerms] exception at index " << i << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[P2PCVSuperCostTerm | buildGaussNewtonTerms] exception at index " << i << ": (unknown)" << std::endl;
            }
        }
    }
}  // namespace finalicp