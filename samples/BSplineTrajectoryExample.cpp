#include <iomanip>
#include <iostream>

#include <liegroupmath.hpp>
#include <slam.hpp>

using namespace finalicp;

/** \brief Example that loads and solves simple bundle adjustment problems */
int main(int argc, char** argv) {
  const double T = 1.0;

  // include/trajectory/time.hpp
  traj::Time knot_spacing(0.4);

  // include/trajectory/bspline/interface.hpp
  traj::bspline::Interface traj(knot_spacing);

  std::vector<std::pair<traj::Time, Eigen::Matrix<double, 6, 1>>> w_iv_inv_meas;

  // 9
  w_iv_inv_meas.emplace_back(traj::Time(0.1 * T), 0.0 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.2 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.3 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.4 * T), 0.6 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.5 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.6 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.7 * T), 0.0 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.8 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.9 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());

  // include/problem/costterm/basecostterm.hpp
  std::vector<BaseCostTerm::Ptr> cost_terms;

  // include/problem/lossfunc/l2lossfunc.hpp
  const auto loss_func = L2LossFunc::MakeShared();

  // include/problem/noisemodel/staticnoisemodel.hpp
  const auto noise_model = StaticNoiseModel<6>::MakeShared(Eigen::Matrix<double, 6, 6>::Identity());

  for (auto& meas : w_iv_inv_meas) {
    // 1 - include/evaluable/vspace/vspaceerrorevaluator.hpp
    // 1.1 - template <int DIM>typename VSpaceErrorEvaluator<DIM>::Ptr vspace_error(
    //        const typename Evaluable<typename VSpaceErrorEvaluator<DIM>::InType>::ConstPtr& v,
    //        const typename VSpaceErrorEvaluator<DIM>::InType& v_meas);
    // 2 - src/trajectory/bspline/interface.cpp
    //        Evaluable<VeloType>::ConstPtr getVelocityInterpolator(const Time& time);
    const auto error_func = vspace::vspace_error<6>(traj.getVelocityInterpolator(meas.first), meas.second);

    // include/problem/costterm/weightleastsqcostterm.hpp
    cost_terms.emplace_back(WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func));
  }

  // include/problem/optimizationproblem.hpp
  OptimizationProblem problem(1);

  // include/trajectory/bspline/interface.hpp
  traj.addStateVariables(problem);

  // include/problem/optimizationproblem.hpp
  for (const auto& cost : cost_terms) problem.addCostTerm(cost);

  // include/solver/gausnewtonsolver.hpp
  GaussNewtonSolver::Params params;

  params.verbose = true;

  // include/solver/gausnewtonsolver.hpp
  GaussNewtonSolver solver(problem, params);
  
  // include/solver/solverbase.hpp
  solver.optimize();

  return 0;


  //step:
  // 1 - src/solver/solverbase.cpp
  // 2 - src/solver/gausnewtonsolver.cpp
  // 3 - src/problem/optimizationproblem.cpp
  // 4 - include/problem/costterm/weightleastsqcostterm.hpp
  // 5 - src/matrixoperator/vector.cpp
}