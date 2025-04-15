cmake_minimum_required(VERSION 3.16)
project(math VERSION 1.1.0)

# Compiler setup
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Enables GNU extensions for gnu++20
add_compile_options(-march=native -O3 -Wall -pedantic)

# Find dependencies
find_package(Eigen3 3.3.7 REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Source files for the lgmath library
set(SOURCE_FILES
  src/r3/operations.cpp
  src/se3/operations.cpp
  src/se3/transformations.cpp
  src/se3/transformationswithcovariance.cpp
  src/commonmath.cpp
  src/so3/operations.cpp
  src/so3/rotations.cpp
)

# Build the lgmath shared library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Define PROJECT_LIBRARY for Config.cmake.in
set(PROJECT_LIBRARY ${PROJECT_NAME})

# Build the test executable
add_executable(se3_tests test/se3test.cpp)
target_link_libraries(se3_tests
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  ${GTEST_MAIN_LIBRARIES}
  pthread
)

add_executable(so3_tests test/so3test.cpp)
target_link_libraries(so3_tests
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  ${GTEST_MAIN_LIBRARIES}
  pthread
)

add_executable(transform_test test/transformtest.cpp)
target_link_libraries(transform_test
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  ${GTEST_MAIN_LIBRARIES}
  pthread
)

add_executable(transformwithcov_test test/transformwithcovariancetests.cpp)
target_link_libraries(transformwithcov_test
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  ${GTEST_MAIN_LIBRARIES}
  pthread
)

add_executable(covariance_test test/covariancetest.cpp)
target_link_libraries(covariance_test
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  ${GTEST_MAIN_LIBRARIES}
  pthread
)

add_executable(rotation_test test/rotationtest.cpp)
target_link_libraries(rotation_test
  ${PROJECT_NAME}
  ${GTEST_LIBRARIES}
  ${GTEST_MAIN_LIBRARIES}
  pthread
)

# Install library and headers
install(
  DIRECTORY include/
  DESTINATION include
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
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
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
add_test(NAME SE3Tests COMMAND se3_tests)
add_test(NAME SO3Tests COMMAND so3_tests)
add_test(NAME TRANSFORMTests COMMAND transform_test)
add_test(NAME TRANSFORMWITHCOVTests COMMAND transformwithcov_test)
add_test(NAME COVARIANCETests COMMAND covariance_test)
add_test(NAME ROTATIONTests COMMAND rotation_test)git add .