cmake_minimum_required(VERSION 3.22)
project(vision_ai_interfaces)

# 指定conda环境Python根目录
set(Python3_ROOT_DIR "/home/qi/miniconda3/envs/rel_env")
set(Python3_FIND_STRATEGY "LOCATION")

# 设置 RPATH 策略，避免运行时库冲突警告
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(${NumPy_INCLUDE_DIR})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ScanPlan.msg"
  "msg/Waypoint.msg"
  "msg/GridPosition.msg"      # 🆕 新增
  "msg/RelativePosition.msg"  # 🆕 新增
  "msg/StitchResult.msg"
  "msg/ImageData.msg"
  "msg/DetectedObject.msg"
  "msg/DetectionResult.msg"
  "srv/PlanScan.srv"
  "srv/ExecuteScan.srv"
  "srv/ProcessStitching.srv"
  "srv/ProcessDetection.srv"
  "srv/SetGripperPosition.srv"
  "srv/SetGripperClose.srv"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()