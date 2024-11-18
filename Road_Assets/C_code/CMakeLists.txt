cmake_minimum_required(VERSION 3.10)

project(my_infer)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
# 消除find cuda警告
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
# ============= tensorrt ============
#include_directories(/home/ubuntu/Documents/dev_env/software/TensorRT-8.6.1.6/include/)
#link_directories(/home/ubuntu/Documents/dev_env/software/TensorRT-8.6.1.6/lib/)
#include_directories(/home/xianyang/Documents/TensorRT-8.6.1.6/include/)
#link_directories(/home/xianyang/Documents/TensorRT-8.6.1.6/lib/)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()
# ============ opencv ============
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# =========== track lib ===========
include_directories(${PROJECT_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${SRCS01})
target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})

# ============= seg lib =============
include_directories(${PROJECT_SOURCE_DIR}/segment/include)
file(GLOB_RECURSE SRCS02 ${PROJECT_SOURCE_DIR}/segment/src/*.cpp ${PROJECT_SOURCE_DIR}/segment/src/*.cu)
cuda_add_library(infer SHARED ${SRCS02})
target_link_libraries(infer nvinfer cudart nvonnxparser ${OpenCV_LIBS})

add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp xytools/tools.hpp  xytools/my_curve.hpp
        )
target_link_libraries(main bytetrack infer)
