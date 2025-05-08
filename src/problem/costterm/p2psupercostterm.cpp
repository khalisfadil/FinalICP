#include <problem/costterm/p2psupercostterm.hpp>

#include <iostream>

namespace finalicp {

    P2PSuperCostTerm::Ptr P2PSuperCostTerm::MakeShared(const Interface::ConstPtr &interface, const Time time1, const Time time2, const Options &options) {
        return std::make_shared<P2PSuperCostTerm>(interface, time1, time2, options);
    }

    double P2PSuperCostTerm::cost() const {

        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto dw1_ = knot1_->acceleration()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();
        const auto dw2_ = knot2_->acceleration()->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto dw1 = dw1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();
        const auto dw2 = dw2_->value();

        // Compute relative pose and velocity transformation
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto J_21_inv_w2 = J_21_inv * w2;
        const auto J_21_inv_curl_dw2 = (-0.5 * math::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

        // const double rinv = 1.0 / options_.r_p2p;
        // const double sqrt_rinv = sqrt(rinv);

        // Sequential processing for small meas_times_ to avoid parallel overhead
        double cost = 0;
        for (unsigned int i = 0; i < meas_times_.size(); ++i) {
            try {
                const double &ts = meas_times_[i];
                const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);

                // Pose interpolation
                const auto &omega = interp_mats_.at(ts).first;
                const auto &lambda = interp_mats_.at(ts).second;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + lambda(0, 2) * dw1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2 + omega(0, 2) * J_21_inv_curl_dw2;
                const math::se3::Transformation T_i1(xi_i1);
                const math::se3::Transformation T_i0 = T_i1 * T1;
                const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

                double cost_i = 0.0;
                for (const int &match_idx : bin_indices) {
                    const auto &p2p_match = p2p_matches_.at(match_idx);
                    const double raw_error = p2p_match.normal.transpose() * (p2p_match.reference - T_mr.block<3, 3>(0, 0) * p2p_match.query - T_mr.block<3, 1>(0, 3));
                    double match_cost = p2p_loss_func_->cost(fabs(raw_error));
                    if (!std::isnan(match_cost)) {cost_i += match_cost;}
                }

                if (!std::isnan(cost_i)) {cost += cost_i;}
            } catch (const std::exception& e) {
                std::cerr << "[P2PSuperCostTerm::cost] exception at timestamp " << meas_times_[i] << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[P2PSuperCostTerm::cost] exception at timestamp " << meas_times_[i] << ": (unknown)" << std::endl;
            }
        }

        return cost;
    }

    void P2PSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
        knot1_->pose()->getRelatedVarKeys(keys);
        knot1_->velocity()->getRelatedVarKeys(keys);
        knot1_->acceleration()->getRelatedVarKeys(keys);
        knot2_->pose()->getRelatedVarKeys(keys);
        knot2_->velocity()->getRelatedVarKeys(keys);
        knot2_->acceleration()->getRelatedVarKeys(keys);
    }

    void P2PSuperCostTerm::initP2PMatches() {
        p2p_match_bins_.clear();
        for (int i = 0; i < (int)p2p_matches_.size(); ++i) {
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
    }

    void P2PSuperCostTerm::initialize_interp_matrices_() {
        const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
        for (const double &time : meas_times_) {
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


            interp_mats_.emplace(time, std::make_pair(omega, lambda));
            }
        }
    }

    void P2PSuperCostTerm::buildGaussNewtonTerms(const StateVector &state_vec, BlockSparseMatrix *approximate_hessian, BlockVector *gradient_vector) const {

        // Retrieve knot states
        using namespace se3;
        using namespace vspace;
        const auto T1_ = knot1_->pose()->forward();
        const auto w1_ = knot1_->velocity()->forward();
        const auto dw1_ = knot1_->acceleration()->forward();
        const auto T2_ = knot2_->pose()->forward();
        const auto w2_ = knot2_->velocity()->forward();
        const auto dw2_ = knot2_->acceleration()->forward();

        const auto T1 = T1_->value();
        const auto w1 = w1_->value();
        const auto dw1 = dw1_->value();
        const auto T2 = T2_->value();
        const auto w2 = w2_->value();
        const auto dw2 = dw2_->value();

        // Compute relative pose and velocity transformation
        const auto xi_21 = (T2 / T1).vec();
        const math::se3::Transformation T_21(xi_21);
        const auto Ad_T_21 = math::se3::tranAd(T_21.matrix());
        const Eigen::Matrix<double, 6, 6> J_21_inv = math::se3::vec2jacinv(xi_21);
        const auto J_21_inv_w2 = J_21_inv * w2;
        const auto J_21_inv_curl_dw2 = (-0.5 * math::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

        // Thread-local accumulators for A and b
        using Matrix36x36 = Eigen::Matrix<double, 36, 36>;
        using Vector36 = Eigen::Matrix<double, 36, 1>;

        // Process measurement times: sequential for small sizes, parallel for large
        Matrix36x36 A = Matrix36x36::Zero();
        Vector36 b = Vector36::Zero();
        for (int i = 0; i < (int)meas_times_.size(); ++i) {
            try {
                const double &ts = meas_times_[i];
                const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);

                // Pose interpolation
                const auto &omega = interp_mats_.at(ts).first;
                const auto &lambda = interp_mats_.at(ts).second;
                const Eigen::Matrix<double, 6, 1> xi_i1 = lambda(0, 1) * w1 + lambda(0, 2) * dw1 + omega(0, 0) * xi_21 + omega(0, 1) * J_21_inv_w2 + omega(0, 2) * J_21_inv_curl_dw2;
                const math::se3::Transformation T_i1(xi_i1);
                const math::se3::Transformation T_i0 = T_i1 * T1;
                const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

                // Pose interpolation Jacobian
                Eigen::Matrix<double, 6, 36> interp_jac = Eigen::Matrix<double, 6, 36>::Zero();
                const Eigen::Matrix<double, 6, 6> J_i1 = math::se3::vec2jac(xi_i1);
                const Eigen::Matrix<double, 6, 6> w = J_i1 * (omega(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() + omega(0, 1) * 0.5 * math::se3::curlyhat(w2) + omega(0, 2) * 0.25 * math::se3::curlyhat(w2) * math::se3::curlyhat(w2) + omega(0, 2) * 0.5 * math::se3::curlyhat(dw2)) * J_21_inv;
                interp_jac.block<6, 6>(0, 0) = -w * Ad_T_21 + T_i1.adjoint(); // T1
                interp_jac.block<6, 6>(0, 6) = lambda(0, 1) * J_i1; // w1
                interp_jac.block<6, 6>(0, 12) = lambda(0, 2) * J_i1; // dw1
                interp_jac.block<6, 6>(0, 18) = w; // T2
                interp_jac.block<6, 6>(0, 24) = omega(0, 1) * J_i1 * J_21_inv + omega(0, 2) * -0.5 * J_i1 * (math::se3::curlyhat(J_21_inv * w2) - math::se3::curlyhat(w2) * J_21_inv); // w2
                interp_jac.block<6, 6>(0, 30) = omega(0, 2) * J_i1 * J_21_inv; // dw2

                // Measurement Jacobians
                Eigen::Matrix<double, 1, 6> Gmeas = Eigen::Matrix<double, 1, 6>::Zero();
                double error = 0.0;

                for (const int &match_idx : bin_indices) {
                    const auto &p2p_match = p2p_matches_.at(match_idx);
                    const double raw_error = p2p_match.normal.transpose() * (p2p_match.reference - T_mr.block<3, 3>(0, 0) * p2p_match.query - T_mr.block<3, 1>(0, 3));
                    const double sqrt_w = sqrt(p2p_loss_func_->weight(fabs(raw_error)));
                    error += sqrt_w * raw_error;
                    Gmeas += sqrt_w * p2p_match.normal.transpose() * (T_mr * math::se3::point2fs(p2p_match.query)).block<3, 6>(0, 0);
                }

                const Eigen::Matrix<double, 1, 36> G = Gmeas * interp_jac;
                A += G.transpose() * G;
                b -= G.transpose() * error;
            } catch (const std::exception& e) {
                std::cerr << "[P2PSuperCostTerm::buildGaussNewtonTerms] exception at timestamp " << meas_times_[i] << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[P2PSuperCostTerm::buildGaussNewtonTerms] exception at timestamp " << meas_times_[i] << ": (unknown)" << std::endl;
            }
        }

        // Determine active variables and extract keys
        std::vector<bool> active;
        active.push_back(knot1_->pose()->active());
        active.push_back(knot1_->velocity()->active());
        active.push_back(knot1_->acceleration()->active());
        active.push_back(knot2_->pose()->active());
        active.push_back(knot2_->velocity()->active());
        active.push_back(knot2_->acceleration()->active());

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

        // Update global Hessian and gradient for active variables
        for (size_t i = 0; i < 6; ++i) {
            try {
                if (!active[i]) continue;
                const auto& key1 = keys[i];
                unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

                // Debug: Print key, Jacobian size, and block index
                // ################################
                std::cout << "[DEBUG] i=" << i << ", key1=" << key1 << ", blkIdx1=" << blkIdx1 << std::endl;
                // ################################

                // Update gradient
                Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

                // Debug: Print gradient term size and norm
                // ################################
                std::cout << "[DEBUG] Gradient term size: (" << newGradTerm.rows() << ", " << newGradTerm.cols() << "), norm: " << newGradTerm.norm() << std::endl;
                // ################################

                gradient_vector->mapAt(blkIdx1) += newGradTerm;

                // Update Hessian (upper triangle)
                for (size_t j = i; j < 6; ++j) {
                    if (!active[j]) continue;
                    const auto& key2 = keys[j];
                    unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

                    // Debug: Print inner loop key and block index
                    // ################################
                    std::cout << "[DEBUG] j=" << j << ", key2=" << key2 << ", blkIdx2=" << blkIdx2 << std::endl;
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
                    std::cout << "[DEBUG] Hessian term (row=" << row << ", col=" << col << ") size: (" << newHessianTerm.rows() << ", " << newHessianTerm.cols() << "), norm: " << newHessianTerm.norm() << std::endl;
                    // ################################

                    // Update Hessian with mutex protection
                    BlockSparseMatrix::BlockRowEntry& entry = approximate_hessian->rowEntryAt(row, col, true);
                    // omp_set_lock(&entry.lock);
                    entry.data += newHessianTerm;
                    // omp_unset_lock(&entry.lock);
                }
            } catch (const std::exception& e) {
                std::cerr << "[P2PCVSuperCostTerm::buildGaussNewtonTerms] exception at index " << i << ": " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[P2PCVSuperCostTerm::buildGaussNewtonTerms] exception at index " << i << ": (unknown)" << std::endl;
            }
        }
    }
}  // namespace finalicp