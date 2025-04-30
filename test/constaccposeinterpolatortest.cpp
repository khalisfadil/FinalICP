#include <gtest/gtest.h>

#include <iostream>
#include <random>

#include <liegroupmath.hpp>
#include <slam.hpp>

#include <trajectory/constacc/accelerationinterpolator.hpp>
#include <trajectory/constacc/poseinterpolator.hpp>
#include <trajectory/constacc/velocityinterpolator.hpp>

#include <trajectory/constvel/poseinterpolator.hpp>
#include <trajectory/constvel/velocityinterpolator.hpp>

#include <trajectory/singer/accelerationinterpolator.hpp>
#include <trajectory/singer/poseinterpolator.hpp>
#include <trajectory/singer/velocityinterpolator.hpp>

#include <problem/costterm/imusupercostterm.hpp>
#include <problem/costterm/preintegratedimucostterm.hpp>
#include <problem/costterm/p2pglobalperturbsupercostterm.hpp>

TEST(ConstAcc, PoseInterpolator) {
  using namespace finalicp::traj::const_acc;
  using namespace finalicp;
  // TODO: try a bunch of random dw_01_in1;
  const Eigen::Matrix<double, 6, 1> dw_01_in1 = -1 * Eigen::Matrix<double, 6, 1>::Ones();
  const Eigen::Matrix<double, 6, 1> dw_02_in2 = dw_01_in1;
  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Zero();
  math::se3::Transformation T_10;

  const double dt = 0.1;

  Eigen::Matrix<double, 6, 1> w_02_in2 = dw_01_in1 * dt;

  // TODO: try different values of tau between 0 and dt
  traj::Time t1(0.);
  traj::Time t2(dt);
  traj::Time tau(dt / 2);

  // Use Magnus expansion to extrapolate pose using constant acceleration
  math::se3::Transformation T_20 = math::se3::Transformation(Eigen::Matrix<double, 6, 1>(w_01_in1 * dt + 0.5 * dw_01_in1 * pow(dt, 2) + (1 / 12) * math::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 3) + (1 / 240) * math::se3::curlyhat(dw_01_in1) * math::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 5))) * T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto dw_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);
  const auto dw_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2);

  const auto knot1 = std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var);
  const auto knot2 = std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var);

  const auto T_q0_eval = PoseInterpolator::MakeShared(tau, knot1, knot2);

  // check the forward pass is what we expect
  {
    std::cout << T_q0_eval->evaluate().matrix() << std::endl;

    math::se3::Transformation T_q0_expected = math::se3::Transformation(Eigen::Matrix<double, 6, 1>(w_01_in1 * (dt / 2) + 0.5 * dw_01_in1 * pow((dt / 2), 2) + (1 / 12) * math::se3::curlyhat(dw_01_in1) * w_01_in1 * pow((dt / 2), 3) + (1 / 240) * math::se3::curlyhat(dw_01_in1) * math::se3::curlyhat(dw_01_in1) * w_01_in1 *  pow((dt / 2), 5))) * T_10;
    EXPECT_LT((T_q0_expected.matrix() - T_q0_eval->evaluate().matrix()).norm(),1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = T_q0_eval->forward();
  T_q0_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 = se3::SE3StateVar::MakeShared(math::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 = se3::SE3StateVar::MakeShared(math::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 = std::make_shared<Variable>(t1, T_10_var_mod1, w_01_in1_var, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(t1, T_10_var_mod2, w_01_in1_var, dw_01_in1_var);

      const auto T_q0_eval_mod1 = PoseInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto T_q0_eval_mod2 = PoseInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) = (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse()).vec() /(2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(t1, T_10_var, w_01_in1_var_mod1, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(t1, T_10_var, w_01_in1_var_mod2, dw_01_in1_var);

      const auto T_q0_eval_mod1 = PoseInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto T_q0_eval_mod2 = PoseInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) = (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse()).vec() /(2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw1
  {
    std::cout << "dw1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_01_in1_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi1);
      const auto dw_01_in1_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod1);
      const auto knot1_mod2 = std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod2);

      const auto T_q0_eval_mod1 = PoseInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto T_q0_eval_mod2 = PoseInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) = (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse()).vec() /(2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 = se3::SE3StateVar::MakeShared(math::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 = se3::SE3StateVar::MakeShared(math::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 = std::make_shared<Variable>(t2, T_20_var_mod1, w_02_in2_var, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(t2, T_20_var_mod2, w_02_in2_var, dw_02_in2_var);

      const auto T_q0_eval_mod1 = PoseInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto T_q0_eval_mod2 = PoseInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) = (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse()).vec() /(2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(t2, T_20_var, w_02_in2_var_mod1, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(t2, T_20_var, w_02_in2_var_mod2, dw_02_in2_var);

      const auto T_q0_eval_mod1 = PoseInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto T_q0_eval_mod2 =PoseInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) = (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse()).vec() /(2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw2
  {
    std::cout << "dw2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_02_in2_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi1);
      const auto dw_02_in2_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod1);
      const auto knot2_mod2 = std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod2);

      const auto T_q0_eval_mod1 = PoseInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto T_q0_eval_mod2 = PoseInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) = (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse()).vec() /(2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}