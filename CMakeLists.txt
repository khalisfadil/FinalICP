cmake_minimum_required(VERSION 3.16)
project(finalicp VERSION 1.1.0 LANGUAGES CXX)

# Compiler setup
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Enables GNU extensions for gnu++20
add_compile_options(-march=native -O3 -Wall -Wextra -pedantic)

# Find dependencies
find_package(Eigen3 3.3.7 REQUIRED)
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)

# Include directories for 3rd-party libraries
set(THIRDPARTY_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lgmath/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/robin/include
)

# Source files for the finalicp library
set(SOURCE_FILES
    src/matrixoperator/matrix.cpp
    src/matrixoperator/matrixbase.cpp
    src/matrixoperator/matrixindexing.cpp
    src/matrixoperator/matrixsparse.cpp
    src/matrixoperator/vector.cpp

    src/evaluable/imu/accerrorevaluator.cpp
    src/evaluable/imu/dmierrorevaluator.cpp
    src/evaluable/imu/gyroerrorevaluator.cpp
    src/evaluable/imu/gyroerrorevaluatorse2.cpp
    src/evaluable/imu/imuerrorevaluator.cpp

    src/evaluable/p2p/p2perrorevaluator.cpp
    src/evaluable/p2p/p2planeerrorevaluator.cpp
    src/evaluable/p2p/p2planeerrorglobalperturbevaluator.cpp
    src/evaluable/p2p/velerrorevaluator.cpp
    src/evaluable/p2p/yawerrorevaluator.cpp
    src/evaluable/p2p/yawvelerrorevaluator.cpp

    src/evaluable/se3/composeevaluator.cpp
    src/evaluable/se3/composeinverseevaluator.cpp
    src/evaluable/se3/composevelocityevaluator.cpp
    src/evaluable/se3/expmapevaluator.cpp
    src/evaluable/se3/inverseevaluator.cpp
    src/evaluable/se3/logmapevaluator.cpp
    src/evaluable/se3/poseinterpolator.cpp
    src/evaluable/se3/se3errorevaluator.cpp
    src/evaluable/se3/se3globalperturbevaluator.cpp
    src/evaluable/se3/se3statevar.cpp
    src/evaluable/se3/se3statevarglobalperturbevaluator.cpp

    src/problem/costterm/gyrosupercostterm.cpp
    src/problem/costterm/imusupercostterm.cpp
    src/problem/costterm/p2pconstvelsupercostterm.cpp
    src/problem/costterm/p2pglobalperturbsupercostterm.cpp
    src/problem/costterm/p2psupercostterm.cpp
    src/problem/costterm/preintegratedimucostterm.cpp
    src/problem/costterm/preintgratedaccelcostterm.cpp
    src/problem/optimizationproblem.cpp
    src/problem/slidingwindowfilter.cpp
    src/problem/statevector.cpp

    src/solver/covariance.cpp
    src/solver/dogleggaussnewtonsolver.cpp
    src/solver/gausnewtonsolver.cpp
    src/solver/gausnewtonsolvernva.cpp
    src/solver/levmarqgaussnewtonsolver.cpp
    src/solver/linesearchgaussnewtonsolver.cpp
    src/solver/solverbase.cpp

    src/trajectory/bspline/interface.cpp
    src/trajectory/bspline/velocityinterpolator.cpp

    src/trajectory/constacc/evaluable/composecurlyhatevaluator.cpp
    src/trajectory/constacc/accelerationextrapolator.cpp
    src/trajectory/constacc/accelerationinterpolator.cpp
    src/trajectory/constacc/interface.cpp
    src/trajectory/constacc/poseinterpolator.cpp
    src/trajectory/constacc/posextrapolator.cpp
    src/trajectory/constacc/priorfactor.cpp
    src/trajectory/constacc/velocityextrapolator.cpp
    src/trajectory/constacc/velocityinterpolator.cpp

    src/trajectory/constvel/evaluable/jinvvelocityevaluator.cpp
    src/trajectory/constvel/evaluable/jvelocityevaluator.cpp
    src/trajectory/constvel/interface.cpp
    src/trajectory/constvel/poseextrapolator.cpp
    src/trajectory/constvel/poseinterpolator.cpp
    src/trajectory/constvel/priorfactor.cpp
    src/trajectory/constvel/velocityinterpolator.cpp

    3rdparty/lgmath/src/r3/operations.cpp
    3rdparty/lgmath/src/se3/operations.cpp
    3rdparty/lgmath/src/se3/transformations.cpp
    3rdparty/lgmath/src/se3/transformationswithcovariance.cpp
    3rdparty/lgmath/src/commonmath.cpp
    3rdparty/lgmath/src/so3/operations.cpp
    3rdparty/lgmath/src/so3/rotations.cpp
)

# Build the finalicp shared library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${EIGEN3_INCLUDE_DIR}
        ${TBB_INCLUDE_DIRS}
    PRIVATE
        ${THIRDPARTY_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Eigen3::Eigen
        TBB::tbb
)

# Build the test executable
# add_executable(pattern_tests test/patterntest.cpp)
# target_include_directories(pattern_tests
#     PRIVATE
#         ${GTEST_INCLUDE_DIRS}
#         ${THIRDPARTY_INCLUDE_DIRS}
# )
# target_link_libraries(pattern_tests
#     PRIVATE
#         ${PROJECT_NAME}
#         GTest::gtest
#         GTest::gtest_main
#         Threads::Threads
# )

# Build the trajectory and matrix test executable
add_executable(const_acc_pose_interpolator_tests test/constaccposeinterpolatortest.cpp)
target_include_directories(const_acc_pose_interpolator_tests
    PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${THIRDPARTY_INCLUDE_DIRS}
)
target_link_libraries(const_acc_pose_interpolator_tests
    PRIVATE
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

# Install library and headers
install(
    DIRECTORY 
        include/matrixoperator/
        include/common/
        include/evaluable/
        include/problem/
        include/solver/
        include/trajectory/
        include/
    DESTINATION 
        include
    FILES_MATCHING PATTERN "*.hpp"
)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install export for find_package support
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)

# Enable testing
enable_testing()
# add_test(NAME PatternTests COMMAND pattern_tests)
add_test(NAME ConstAccPoseInterpolatorTest COMMAND const_acc_pose_interpolator_tests)