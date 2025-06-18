cmake_minimum_required(VERSION 3.5)
project(localizer)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Ofast)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(geneticLocaliser src/genetic.cpp)
ament_target_dependencies(geneticLocaliser rclcpp std_msgs OpenCV)

install(TARGETS
  geneticLocaliser
  DESTINATION lib/${PROJECT_NAME})

ament_package()