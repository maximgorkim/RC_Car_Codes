# Plain CMake (not a colcon package, see COLCON_IGNORE): builds the closed-loop stack
# simulator from the ROS-free cores of rc_localization and mpc_tracking_controller.
#   cmake -S tools/stack_sim -B build/stack_sim -DCMAKE_BUILD_TYPE=Release && cmake --build build/stack_sim
cmake_minimum_required(VERSION 3.8)
project(stack_sim CXX)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -O2)
find_package(Eigen3 REQUIRED)
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
add_executable(stack_sim stack_sim.cpp
  ${SRC}/rc_localization/src/ekf_core.cpp
  ${SRC}/mpc_tracking_controller/src/path_geometry.cpp
  ${SRC}/mpc_tracking_controller/src/tracking_mpc.cpp)
target_include_directories(stack_sim PRIVATE
  ${SRC}/rc_localization/include ${SRC}/mpc_tracking_controller/include ${EIGEN3_INCLUDE_DIR})
