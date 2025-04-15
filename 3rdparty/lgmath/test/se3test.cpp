#include <gtest/gtest.h>

#include <math.h>
#include <iomanip>
#include <ios>
#include <iostream>

#include <Eigen/Dense>
#include <commonmath.hpp>
#include <se3/operations.hpp>
#include <so3/operations.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////
///
/// UNIT TESTS OF SE(3) MATH
///
/////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(3) hat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(math, Test4x4HatFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 6, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 4, 4> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 4> mat;
    mat << 0.0, -trueVecs.at(i)[5], trueVecs.at(i)[4], trueVecs.at(i)[0],
        trueVecs.at(i)[5], 0.0, -trueVecs.at(i)[3], trueVecs.at(i)[1],
        -trueVecs.at(i)[4], trueVecs.at(i)[3], 0.0, trueVecs.at(i)[2], 0.0, 0.0,
        0.0, 0.0;
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 4> testMat = math::se3::hat(trueVecs.at(i));
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(math::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of SE(3) curlyhat function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(math, TestCurlyHatFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 6, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 6, 6> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 6> mat;
    mat << 0.0, -trueVecs.at(i)[5], trueVecs.at(i)[4], 0.0, -trueVecs.at(i)[2],
        trueVecs.at(i)[1], trueVecs.at(i)[5], 0.0, -trueVecs.at(i)[3],
        trueVecs.at(i)[2], 0.0, -trueVecs.at(i)[0], -trueVecs.at(i)[4],
        trueVecs.at(i)[3], 0.0, -trueVecs.at(i)[1], trueVecs.at(i)[0], 0.0, 0.0,
        0.0, 0.0, 0.0, -trueVecs.at(i)[5], trueVecs.at(i)[4], 0.0, 0.0, 0.0,
        trueVecs.at(i)[5], 0.0, -trueVecs.at(i)[3], 0.0, 0.0, 0.0,
        -trueVecs.at(i)[4], trueVecs.at(i)[3], 0.0;
    trueMats.push_back(mat);
  }

  // Test the function
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 6> testMat = math::se3::curlyhat(trueVecs.at(i));
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(math::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of homogeneous point to 4x6 matrix function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(math, TestPointTo4x6MatrixFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 4, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 4, 6> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 6> mat;
    mat << trueVecs.at(i)[3], 0.0, 0.0, 0.0, trueVecs.at(i)[2],
        -trueVecs.at(i)[1], 0.0, trueVecs.at(i)[3], 0.0, -trueVecs.at(i)[2],
        0.0, trueVecs.at(i)[0], 0.0, 0.0, trueVecs.at(i)[3], trueVecs.at(i)[1],
        -trueVecs.at(i)[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 4, 6> testMat =
        math::se3::point2fs(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(math::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of homogeneous point to 6x4 matrix function
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(math, TestPointTo6x4MatrixFunction) {
  // Number of random tests
  const unsigned numTests = 20;

  // Add vectors to be tested - random
  std::vector<Eigen::Matrix<double, 4, 1> > trueVecs;
  for (unsigned i = 0; i < numTests; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 4, 1>::Random());
  }

  // Setup truth matrices
  std::vector<Eigen::Matrix<double, 6, 4> > trueMats;
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 4> mat;
    mat << 0.0, 0.0, 0.0, trueVecs.at(i)[0], 0.0, 0.0, 0.0, trueVecs.at(i)[1],
        0.0, 0.0, 0.0, trueVecs.at(i)[2], 0.0, trueVecs.at(i)[2],
        -trueVecs.at(i)[1], 0.0, -trueVecs.at(i)[2], 0.0, trueVecs.at(i)[0],
        0.0, trueVecs.at(i)[1], -trueVecs.at(i)[0], 0.0, 0.0;
    trueMats.push_back(mat);
  }

  // Test the 3x1 function with scaling param
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 4> testMat =
        math::se3::point2sf(trueVecs.at(i).head<3>(), trueVecs.at(i)[3]);
    std::cout << "true: " << trueMats.at(i) << std::endl;
    std::cout << "func: " << testMat << std::endl;
    EXPECT_TRUE(math::common::nearEqual(trueMats.at(i), testMat, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential functions: vec2tran and tran2vec
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(math, CompareAnalyticalAndNumericVec2Tran) {
  // Add vectors to be tested
  std::vector<Eigen::Matrix<double, 6, 1> > trueVecs;
  Eigen::Matrix<double, 6, 1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, math::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -math::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5 * math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5 * math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * math::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc matrices
  std::vector<Eigen::Matrix<double, 4, 4> > analyticTrans;
  for (unsigned i = 0; i < numTests; i++) {
    analyticTrans.push_back(math::se3::vec2tran(trueVecs.at(i)));
  }

  // Compare analytical and numeric result
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 4, 4> numericTran =
          math::se3::vec2tran(trueVecs.at(i), 20);
      std::cout << "ana: " << analyticTrans.at(i) << std::endl;
      std::cout << "num: " << numericTran << std::endl;
      EXPECT_TRUE(
          math::common::nearEqual(analyticTrans.at(i), numericTran, 1e-6));
    }
  }

  // Test rot2vec
  {
    for (unsigned i = 0; i < numTests; i++) {
      Eigen::Matrix<double, 6, 1> testVec =
          math::se3::tran2vec(analyticTrans.at(i));
      std::cout << "true: " << trueVecs.at(i) << std::endl;
      std::cout << "func: " << testVec << std::endl;
      EXPECT_TRUE(
          math::common::nearEqualLieAlg(trueVecs.at(i), testVec, 1e-6));
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of exponential jacobians: vec2jac and vec2jacinv
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(math, CompareAnalyticalJacobInvAndNumericCounterpartsInSE3) {
  // Add vectors to be tested
  std::vector<Eigen::Matrix<double, 6, 1> > trueVecs;
  Eigen::Matrix<double, 6, 1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, math::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -math::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5 * math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5 * math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * math::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Calc analytical matrices
  std::vector<Eigen::Matrix<double, 6, 6> > analyticJacs;
  std::vector<Eigen::Matrix<double, 6, 6> > analyticJacInvs;
  for (unsigned i = 0; i < numTests; i++) {
    analyticJacs.push_back(math::se3::vec2jac(trueVecs.at(i)));
    analyticJacInvs.push_back(math::se3::vec2jacinv(trueVecs.at(i)));
  }

  // Compare inversed analytical and analytical inverse
  for (unsigned i = 0; i < numTests; i++) {
    std::cout << "ana: " << analyticJacs.at(i) << std::endl;
    std::cout << "num: " << analyticJacInvs.at(i) << std::endl;
    EXPECT_TRUE(math::common::nearEqual(analyticJacs.at(i).inverse(),
                                          analyticJacInvs.at(i), 1e-6));
  }

  // Compare analytical and 'numerical' jacobian
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 6> numericJac =
        math::se3::vec2jac(trueVecs.at(i), 20);
    std::cout << "ana: " << analyticJacs.at(i) << std::endl;
    std::cout << "num: " << numericJac << std::endl;
    EXPECT_TRUE(
        math::common::nearEqual(analyticJacs.at(i), numericJac, 1e-6));
  }

  // Compare analytical and 'numerical' jacobian inverses
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 6> numericJac =
        math::se3::vec2jacinv(trueVecs.at(i), 20);
    std::cout << "ana: " << analyticJacInvs.at(i) << std::endl;
    std::cout << "num: " << numericJac << std::endl;
    EXPECT_TRUE(
        math::common::nearEqual(analyticJacInvs.at(i), numericJac, 1e-6));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General test of adjoint tranformation identity, Ad(T(v)) = I +
/// curlyhat(v)*J(v)
/////////////////////////////////////////////////////////////////////////////////////////////
TEST(math, TestIdentityAdTvEqualIPlusCurlyHatvTimesJv) {
  // Add vectors to be tested
  std::vector<Eigen::Matrix<double, 6, 1> > trueVecs;
  Eigen::Matrix<double, 6, 1> temp;
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, math::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, -math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, -math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, -math::constants::PI;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.5 * math::constants::PI, 0.0, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.5 * math::constants::PI, 0.0;
  trueVecs.push_back(temp);
  temp << 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * math::constants::PI;
  trueVecs.push_back(temp);
  const unsigned numRand = 20;
  for (unsigned i = 0; i < numRand; i++) {
    trueVecs.push_back(Eigen::Matrix<double, 6, 1>::Random());
  }

  // Get number of tests
  const unsigned numTests = trueVecs.size();

  // Test Identity
  for (unsigned i = 0; i < numTests; i++) {
    Eigen::Matrix<double, 6, 6> lhs =
        math::se3::tranAd(math::se3::vec2tran(trueVecs.at(i)));
    Eigen::Matrix<double, 6, 6> rhs = Eigen::Matrix<double, 6, 6>::Identity() +
                                      math::se3::curlyhat(trueVecs.at(i)) *
                                          math::se3::vec2jac(trueVecs.at(i));
    std::cout << "lhs: " << lhs << std::endl;
    std::cout << "rhs: " << rhs << std::endl;
    EXPECT_TRUE(math::common::nearEqual(lhs, rhs, 1e-6));
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
