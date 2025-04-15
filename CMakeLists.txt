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
add_executable(pattern_tests test/patterntest.cpp)
target_include_directories(pattern_tests
    PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ${THIRDPARTY_INCLUDE_DIRS}
)
target_link_libraries(pattern_tests
    PRIVATE
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

# Install library and headers
install(
    DIRECTORY include/matrixoperator/
    DESTINATION include/matrixoperator
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
add_test(NAME PatternTests COMMAND pattern_tests)