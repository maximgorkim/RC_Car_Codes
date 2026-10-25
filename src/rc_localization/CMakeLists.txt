cmake_minimum_required(VERSION 3.8)
project(rc_localization)

# Default to an optimized build: without this, colcon builds with no
# optimization at all, which is far too slow on the Jetson Xavier.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(vesc_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR})

# ROS-free filter core (shared by the node and the unit tests)
add_library(ekf_core STATIC src/ekf_core.cpp)
set_target_properties(ekf_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(twist_estimator src/twist_estimator.cpp)
ament_target_dependencies(twist_estimator
  rclcpp geometry_msgs sensor_msgs vesc_msgs
)

add_executable(ekf_localizer src/ekf_localizer_node.cpp)
target_link_libraries(ekf_localizer ekf_core)
ament_target_dependencies(ekf_localizer
  rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs tf2_ros
)

# Offline tuning tool: runs the filter on a recorded sensor stream
add_executable(ekf_replay tools/ekf_replay.cpp)
target_link_libraries(ekf_replay ekf_core)

install(TARGETS
  twist_estimator
  ekf_localizer
  ekf_replay
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/localization_monitor.py
  scripts/sim_vesc_bridge.py
  scripts/live_status.py
  scripts/scan_for_slam.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ekf_core test/test_ekf_core.cpp)
  target_link_libraries(test_ekf_core ekf_core)
endif()

ament_package()
