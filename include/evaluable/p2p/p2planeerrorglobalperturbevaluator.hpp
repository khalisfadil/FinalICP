#pragma once

#include <Eigen/Core>

#include <liegroupmath.hpp>
#include <slam.hpp>

#include <trajectory/time.hpp>

namespace finalicp {
    namespace p2p {
        class P2PlaneErrorGlobalPerturbEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
            public:
                using Ptr = std::shared_ptr<P2PlaneErrorGlobalPerturbEvaluator>;
                using ConstPtr = std::shared_ptr<const P2PlaneErrorGlobalPerturbEvaluator>;

                using InType = math::se3::Transformation;
                using OutType = Eigen::Matrix<double, 1, 1>;
                using Time = traj::Time;

                //Factory method to create an instance.
                static Ptr MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                                            const Eigen::Vector3d &reference,
                                            const Eigen::Vector3d &query,
                                            const Eigen::Vector3d &normal);

                //Constructor.
                P2PlaneErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                                            const Eigen::Vector3d &reference,
                                            const Eigen::Vector3d &query,
                                            const Eigen::Vector3d &normal);

                //Checks if the DMI error is influenced by active state variables.
                bool active() const override;

                //Collects state variable keys that influence this evaluator.
                void getRelatedVarKeys(KeySet &keys) const override;

                //Computes the DMI error.
                OutType value() const override;

                //Forward evaluation of DMI error.
                Node<OutType>::Ptr forward() const override;

                //Computes Jacobians for the DMI error.
                void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node, Jacobians &jacs) const override;

                //Sets the gravity vector.
                void setTime(Time time) {
                    time_ = time;
                    time_init_ = true;
                };

                Time getTime() const {
                    if (time_init_)
                    return time_;
                    else
                    throw std::runtime_error("P2P measurement time was not initialized");
                }

                Eigen::Matrix<double, 3, 6> getJacobianPose() const;
                
            private:
                const Evaluable<InType>::ConstPtr T_rq_;
                const Eigen::Vector3d reference_;
                const Eigen::Vector3d query_;
                const Eigen::Vector3d normal_;
                bool time_init_ = false;
                Time time_;
        };

        //Factory function for creating a GyroErrorEvaluator.
        P2PlaneErrorGlobalPerturbEvaluator::Ptr p2planeGlobalError(const Evaluable<P2PlaneErrorGlobalPerturbEvaluator::InType>::ConstPtr &T_rq,
                                                                    const Eigen::Vector3d &reference, const Eigen::Vector3d &query, const Eigen::Vector3d &normal);
     }  // namespace p2p
}  // namespace finalicp
