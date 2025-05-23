#include <iostream>

#include <liegroupmath.hpp>
#include <slam.hpp>


using namespace finalicp;
using namespace finalicp::traj;

/** \brief Structure to store trajectory state variables */
struct TrajStateVar {
  Time time;
  se3::SE3StateVar::Ptr pose;
  vspace::VSpaceStateVar<6>::Ptr velocity;
};

/** \brief Example that uses a constant-velocity prior over a trajectory. */
int main(int argc, char** argv) {
  ///
  /// Setup Problem
  ///

  // Number of state times
  unsigned int numPoses = 100;

  // Setup velocity prior
  Eigen::Matrix<double, 6, 1> velocityPrior;
  double v_x = -1.0;
  double omega_z = 0.01;
  velocityPrior << v_x, 0.0, 0.0, 0.0, 0.0, omega_z;

  // Calculate time to do one circle
  double totalTime = 2.0 * M_PI / omega_z;

  // Calculate time between states
  double delT = totalTime / (numPoses - 1);

  // Smoothing factor diagonal
  Eigen::Array<double, 1, 6> Qc_diag;
  Qc_diag << 1.0, 0.001, 0.001, 0.001, 0.001, 1.0;

  //
  // Setup initial conditions
  //

  // Pose
  Eigen::Matrix<double, 6, 1> initPoseVec;
  initPoseVec << 1, 2, 3, 4, 5, 6;
  math::se3::Transformation initPose(initPoseVec);

  // Zero velocity
  Eigen::Matrix<double, 6, 1> initVelocity;
  initVelocity.setZero();

  ///
  /// Setup States
  ///

  // Steam state variables
  std::vector<TrajStateVar> states;

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 0; i < numPoses; i++) {
    TrajStateVar temp;
    temp.time = Time(i * delT);
    temp.pose = se3::SE3StateVar::MakeShared(initPose);
    temp.velocity = vspace::VSpaceStateVar<6>::MakeShared(initVelocity);
    states.push_back(temp);
  }

  // Setup Trajectory
  const_vel::Interface traj(Qc_diag);
  for (const auto& state : states)
    traj.add(state.time, state.pose, state.velocity);

  // Add priors (alternatively, we could lock pose variable)
  traj.addPosePrior(Time(0.0), initPose,
                    Eigen::Matrix<double, 6, 6>::Identity());
  traj.addVelocityPrior(Time(0.0), velocityPrior,
                        Eigen::Matrix<double, 6, 6>::Identity());

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  finalicp::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }

  // Add cost terms
  traj.addPriorCostTerms(problem);

  ///
  /// Setup Solver and Optimize
  ///
  finalicp::DoglegGaussNewtonSolver::Params params;
  params.verbose = true;
  finalicp::DoglegGaussNewtonSolver solver(problem, params);

  // Optimize
  solver.optimize();

  // Get velocity at interpolated time
  Time new_time(states.at(0).time.seconds() + 0.5 * delT);
  auto curr_vel = traj.getVelocityInterpolator(new_time)->evaluate();

  ///
  /// Print results
  ///
  // clang-format off
    //   std::cout << std::endl
    //             << "First Pose:                  " << states.at(0).pose->value()
    //             << "Second Pose:                 " << states.at(1).pose->value()
    //             << "Last Pose (full circle):     " << states.back().pose->value()
    //             << "First Vel:                   " << states.at(0).velocity->value().transpose() << std::endl
    //             << "Second Vel:                  " << states.at(1).velocity->value().transpose() << std::endl
    //             << "Last Vel:                    " << states.back().velocity->value().transpose() << std::endl
    //             << "Interp. Vel (t=t0+0.5*delT): " << curr_vel.transpose() << std::endl;
  // clang-format on

  return 0;
}
