cmake_minimum_required(VERSION 3.10)

project(qd VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add source files
set(SOURCES
    src/quantum_dynamics.cpp
)

# Add header files
set(HEADERS
    include/quantum_dynamics.hpp
)

# Create library
target_sources(qd PRIVATE ${SOURCES})

# Add include directories
target_include_directories(qd PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add Eigen3 as a dependency
target_link_libraries(qd PUBLIC Eigen3::Eigen)

# Add example executable
add_executable(qd_example examples/example.cpp)
target_link_libraries(qd_example PRIVATE qd)
