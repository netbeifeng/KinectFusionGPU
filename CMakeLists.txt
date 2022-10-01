cmake_minimum_required(VERSION 3.7)

SET(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
SET(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE PATH "Path to your vcpkg")

PROJECT(KinectFusionGPU)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(PCL CONFIG REQUIRED)
find_package(OpenCL REQUIRED)

SET(HEADERS
    DataLoader.hpp
    GPUProgramLoader.hpp
    PointCloudGenerator.hpp
    SurfaceMeasurement.hpp
    PoseEstimator.hpp
    SurfaceReconstructor.hpp
    SurfacePredictor.hpp
    Utils.h
)

SET(SOURCES 
    main.cpp
)

add_executable(KinectFusionGPU ${HEADERS} ${SOURCES})

LINK_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
target_include_directories(KinectFusionGPU PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(KinectFusionGPU PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(KinectFusionGPU OpenCL::OpenCL)
target_link_libraries(KinectFusionGPU ${OpenCV_LIBS})
target_link_libraries(KinectFusionGPU ${PCL_LIBRARIES})
target_link_libraries(KinectFusionGPU Eigen3::Eigen)