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
        std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | cost]  ###################### START ####################. \n" << std::endl;
        std::cout << "[001# P2PCVSuperCostTerm DEBUG | cost] Calculating cost for " << p2p_matches_.size() << " matches across " << meas_times_.size() << " timestamps..." << std::endl;
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
                             std::cerr << "[002# P2PCVSuperCostTerm DEBUG | cost] CRITICAL: Raw P2P error is non-finite!" << std::endl;
                        } else {
                            std::cout << "[003# P2PCVSuperCostTerm DEBUG | cost] First match (t=" << std::fixed << std::setprecision(4) << ts << "): raw_error = " << raw_error << std::endl;
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
#ifdef DEBUG
        std::cout << "[004# P2PCVSuperCostTerm DEBUG | cost] Total p2p super cost contribution: " << cost << std::endl;
        std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | cost]  ###################### END ####################. \n" << std::endl;
#endif
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
        std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | initP2PMatches]  ###################### START ####################. \n" << std::endl;
        std::cout << "[001# P2PCVSuperCostTerm DEBUG | initP2PMatches] SIZE of p2p_matches_ " << p2p_matches_.size() << " matches by timestamp." << std::endl;
        // [DEBUG] Check if keypoint coordinates are finite before association
        bool matches_are_finite = true;
        for (size_t i = 0; i < p2p_matches_.size(); ++i) {
            if (!p2p_matches_[i].reference.allFinite() || !p2p_matches_[i].normal.allFinite() || !p2p_matches_[i].query.allFinite()) {
                std::cout << "[002# P2PCVSuperCostTerm | initP2PMatches | ts " << p2p_matches_[i].timestamp << "] " << "CRITICAL: p2p_matches_ " << i << " is NOT finite before association!" << std::endl;
                matches_are_finite = false;
                break;
            }
        }
        if (matches_are_finite) {
            std::cout << "[003# P2PCVSuperCostTerm | initP2PMatches ] " << "All p2p_matches_ are finite before association." << std::endl;
        }
#endif
        p2p_match_bins_.clear();
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
      
        meas_times_.clear();
        for (auto it = p2p_match_bins_.begin(); it != p2p_match_bins_.end(); it++) {
            meas_times_.push_back(it->first);
        }
        initialize_interp_matrices_();
#ifdef DEBUG
        std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | initP2PMatches]  ###################### END ####################. \n" << std::endl;
#endif
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    void P2PCVSuperCostTerm::initialize_interp_matrices_() {
#ifdef DEBUG
        std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_]  ###################### START ####################. \n" << std::endl;
        std::cout << "[001# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_] SIZE of meas_times_ " << meas_times_.size() << std::endl;
#endif
        interp_mats_.clear();
        const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
        for (const double &time : meas_times_) {
            if (interp_mats_.find(time) == interp_mats_.end()) { // if time does not exits
                const double tau = (Time(time) - time1_).seconds();
#ifdef DEBUG
                if (tau < 1e-6) {
                std::cerr << "[002# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_] CRITICAL: Small tau=" << tau << std::endl;
#endif

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
#ifdef DEBUG
                if (!Q_tau.allFinite()) {
                    std::cerr << "[003# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_ | ts: " << time << "] CRITICAL: Aggregated Q_tau is non-finite!" << std::endl;
                } 
                if (!Tran_kappa.allFinite()) {
                    std::cerr << "[004# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_ | ts: " << time << "] CRITICAL: Aggregated Tran_kappa is non-finite!" << std::endl;
                } 
                if (!Tran_tau.allFinite()) {
                    std::cerr << "[005# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_ | ts: " << time << "] CRITICAL: Aggregated Tran_tau is non-finite!" << std::endl;
                } 
                if (!omega12.allFinite()) {
                    std::cerr << "[006# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_ | ts: " << time << "] CRITICAL: Aggregated omega12 is non-finite!" << std::endl;
                } 
                if (!lambda12.allFinite()) {
                    std::cerr << "[007# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_ | ts: " << time << "] CRITICAL: Aggregated lambda12 is non-finite!" << std::endl;
                } 
                if (!omega.allFinite()) {
                    std::cerr << "[008# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_ | ts: " << time << "] CRITICAL: Aggregated omega is non-finite!" << std::endl;
                } 
                if (!lambda.allFinite()) {
                    std::cerr << "[009# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_ | ts: " << time << "] CRITICAL: Aggregated lambda is non-finite!" << std::endl;
                }
                std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | initialize_interp_matrices_]  ###################### END ####################. \n" << std::endl; 
#endif
            }
        }
    }

    // ##########################################
    // getRelatedVarKeys
    // ##########################################

    void P2PCVSuperCostTerm::buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const {
#ifdef DEBUG 
        std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms]  ###################### START ####################. \n" << std::endl; 
        std::cout << "[001# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms] SIZE of meas_times_ " << meas_times_.size() << std::endl;
#endif
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
        for (int i = 0; i < static_cast<int>(meas_times_.size()); ++i) {
            try {
                const double &ts = meas_times_[i];
                const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);

                // Pose interpolation
                const auto &omega = interp_mats_.at(ts).first;
                const auto &lambda = interp_mats_.at(ts).second;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2;
                const math::se3::Transformation T_i1(xi_i1);
                const math::se3::Transformation T_i0 = T_i1 * T1;
                const Eigen::Matrix4d Tb2m = T_i0.inverse().matrix();

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
                    const double raw_error = p2p_match.normal.transpose() * (p2p_match.reference - Tb2m.block<3, 3>(0, 0) * p2p_match.query - Tb2m.block<3, 1>(0, 3));
                    const double sqrt_w = sqrt(p2p_loss_func_->weight(fabs(raw_error)));
                    error += sqrt_w * sqrt_rinv * raw_error;
                    Gmeas += sqrt_w * sqrt_rinv * p2p_match.normal.transpose() * (Tb2m * math::se3::point2fs(p2p_match.query)).block<3, 6>(0, 0);
                }

                const Eigen::Matrix<double, 1, 24> G = Gmeas * interp_jac;
                A += G.transpose() * G;
                b -= G.transpose() * error;
#ifdef DEBUG
            if (!xi_i1.allFinite()) {
                std::cerr << "[002# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: xi_i1 is non-finite!" << std::endl;
            }
            double theta = xi_i1.head<3>().norm();
            if (theta < 1e-6) {
                std::cerr << "[003# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] WARNING: Small rotation theta=" << theta << std::endl;
            }
            if (!omega.allFinite()) {
                std::cerr << "[004# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: Aggregated omega is non-finite!" << std::endl;
            }
            if (!lambda.allFinite()) {
                std::cerr << "[005# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: Aggregated lambda is non-finite!" << std::endl;
            }
            if (!J_i1.allFinite()) {
                std::cerr << "[006# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: Aggregated J_i1 is non-finite!" << std::endl;
            }  
            if (!w.allFinite()) {
                std::cerr << "[007# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: Aggregated w is non-finite!" << std::endl;
            }  
            if (!Gmeas.allFinite()) {
                std::cerr << "[008# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: Aggregated measurement Jacobian (Gmeas) is non-finite!" << std::endl;
            } 
            if (!interp_jac.allFinite()) {
                std::cerr << "[009# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: Pose interpolation Jacobian (interp_jac) is non-finite!" << std::endl;
            }
            if (!Gmeas.allFinite()) {
                std::cerr << "[010# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms | iter: " << i << "] CRITICAL: Aggregated measurement Jacobian (Gmeas) is non-finite!" << std::endl;
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
            std::cerr << "[011# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms] CRITICAL: Accumulated local Hessian (A) or Gradient (b) is non-finite!" << std::endl;
        } else {
             std::cout << "[012# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms] Accumulated local Hessian norm: " << A.norm() << ", Gradient norm: " << b.norm() << std::endl;
        }
#endif

        // Determine active variables and extract keys
        std::vector<bool> active;
        std::vector<StateKey> keys;

        {
            const auto T1node = std::static_pointer_cast<Node<PoseType>>(T1_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot1_->pose()->backward(lhs, T1node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        }

        {
            const auto w1node = std::static_pointer_cast<Node<VelType>>(w1_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot1_->velocity()->backward(lhs, w1node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        } 

        {
            const auto T2node = std::static_pointer_cast<Node<PoseType>>(T2_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot2_->pose()->backward(lhs, T2node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        } 

        {
            const auto w2node = std::static_pointer_cast<Node<VelType>>(w2_);
            Jacobians jacs;
            Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
            knot2_->velocity()->backward(lhs, w2node, jacs);
            const auto jacmap = jacs.get();
            assert(jacmap.size() == 1);
            for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
                keys.push_back(it->first);
            }
        } 

        active.push_back(knot1_->pose()->active());
        active.push_back(knot1_->velocity()->active());
        active.push_back(knot2_->pose()->active());
        active.push_back(knot2_->velocity()->active());

        // Update global Hessian and gradient for active variables
        for (size_t i = 0; i < 4; ++i) {
            try {
                if (!active[i]) continue;
                const auto& key1 = keys[i];
                unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

                // Update gradient
                Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

                { gradient_vector->mapAt(blkIdx1) += newGradTerm; }

                // Update Hessian (upper triangle)
                for (size_t j = i; j < 4; ++j) {
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
                    BlockSparseMatrix::BlockRowEntry& entry = approximate_hessian->rowEntryAt(row, col, true);
                    entry.data += newHessianTerm;
                }
            } catch (const std::exception& e) {
                std::cerr << "[P2PCVSuperCostTerm | buildGaussNewtonTerms] exception at index " << i << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[P2PCVSuperCostTerm | buildGaussNewtonTerms] exception at index " << i << ": (unknown)" << std::endl;
            }
        }
#ifdef DEBUG
        std::cout << "\n[000# P2PCVSuperCostTerm DEBUG | buildGaussNewtonTerms]  ###################### END ####################. \n" << std::endl;
#endif
    }
}  // namespace finalicp