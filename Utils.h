#pragma once
#ifndef _UTILS_H
#define _UTILS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <CL/cl.h>
#include <cmath>
#include <regex>
#include <chrono>
//#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/core/utils/logger.hpp>
//#include <pcl/io/pcd_io.h>
//#include <pcl/io/ply_io.h>
//#include <pcl/point_types.h>

#define CL_TARGET_OPENCL_VERSION 220
#define NVIDIA_GPU 

#define DATA_PATH "../data/rgbd_dataset_freiburg1_xyz/"
#define VOLUME_RES 256

#ifdef NVIDIA_GPU
#define WARP_SIZE 32
#else
#define WARP_SIZE 64
#endif

#define ICP_MAX_ITER 20

#define CX  318.6f
#define CY  255.3f 
#define FX  517.3f
#define FY  516.5f

#define FRAME_HEIGHT 480
#define FRAME_WIDTH 640
#define RGB_CHANNEL 3
#define DEPTH_CHANNEL 1

#define INITIAL_X 0.15
#define INITIAL_Y 0.10
#define INITIAL_Z -2.0

#define WORK_SIZE_X ((FRAME_WIDTH + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE
#define WORK_SIZE_Y ((FRAME_HEIGHT + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE

typedef struct
{
    cl_float3 r1; //Row 1
    cl_float3 r2; //Row 2
    cl_float3 r3; //Row 3
} GPUMatrix3f;

typedef struct
{
    cl_float4 r1; //Row 1
    cl_float4 r2; //Row 2
    cl_float4 r3; //Row 3
    cl_float4 r4; //Row 4
} GPUMatrix4f;

struct Frame {
    int lineId;
    std::string id; // CameraPos.TimeStamp
    Eigen::Matrix4f pose;
    std::string depthId;
    std::string depthPath;
    std::string rgbId;
    std::string rgbPath;


    cv::Mat getDepthFrame() {
        cv::Mat depthFrame = cv::imread(DATA_PATH + depthPath, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
        depthFrame.convertTo(depthFrame, CV_32FC1, (1.0 / 5000.0));
        return depthFrame;
    }

    cv::Mat getRGBFrame() {
        cv::Mat rgbFrame = cv::imread(DATA_PATH + rgbPath);
        rgbFrame.convertTo(rgbFrame, CV_32FC3, 1.0f / 255.f);
        return rgbFrame;
    }
};

// Get Intrinsic In Eigen Matrix
inline Eigen::Matrix3f getIntrinsic() {
    Eigen::Matrix3f intrinsic;
    intrinsic << FX, 0.0, CX,
                0.0, FY, CY,
                0.0, 0.0, 1.0;
    return intrinsic;
}

// Get wrapped working Size 
inline size_t* getWorkSize() {
    size_t size[3] = { WORK_SIZE_X , WORK_SIZE_Y, 1 };
    return size;
}

// Get Intrinsic in GPU data
inline GPUMatrix3f getIntrinsicGPU() {
    GPUMatrix3f result;
    result.r1.x = FX;
    result.r1.y = 0.0;
    result.r1.z = CX;

    result.r2.x = 0.0;
    result.r2.y = FY;
    result.r2.z = CY;

    result.r3.x = 0.0;
    result.r3.y = 0.0;
    result.r3.z = 1.0;
    return result;
}

inline Eigen::Matrix4f makeTranslation() {
    Eigen::Matrix4f translation;
    translation << 1.0, 0.0, 0.0, INITIAL_X,
                 0.0, 1.0, 0.0, INITIAL_Y,
                 0.0, 0.0, 1.0, INITIAL_Z,
                 0.0, 0.0, 0.0, 1.0;
    
    return translation;
}

// Convert from Eigen To GPUMatrix3f
inline GPUMatrix3f toGPUMatrix3f(Eigen::Matrix3f eigenM)
{
    GPUMatrix3f result;
    result.r1.x = eigenM(0, 0);
    result.r1.y = eigenM(0, 1);
    result.r1.z = eigenM(0, 2);

    result.r2.x = eigenM(1, 0);
    result.r2.y = eigenM(1, 1);
    result.r2.z = eigenM(1, 2);

    result.r3.x = eigenM(2, 0);
    result.r3.y = eigenM(2, 1);
    result.r3.z = eigenM(2, 2);

    return result;
}

// Convert from Eigen To GPUMatrix4f
inline GPUMatrix4f toGPUMatrix4f(Eigen::Matrix4f eigenM)
{
    GPUMatrix4f result;
    result.r1.s[0] = eigenM(0, 0);
    result.r1.s[1] = eigenM(0, 1);
    result.r1.s[2] = eigenM(0, 2);
    result.r1.s[3] = eigenM(0, 3);

    result.r2.s[0] = eigenM(1, 0);
    result.r2.s[1] = eigenM(1, 1);
    result.r2.s[2] = eigenM(1, 2);
    result.r2.s[3] = eigenM(1, 3);

    result.r3.s[0] = eigenM(2, 0);
    result.r3.s[1] = eigenM(2, 1);
    result.r3.s[2] = eigenM(2, 2);
    result.r3.s[3] = eigenM(2, 3);

    result.r4.s[0] = eigenM(3, 0);
    result.r4.s[1] = eigenM(3, 1);
    result.r4.s[2] = eigenM(3, 2);
    result.r4.s[3] = eigenM(3, 3);
    return result;
}

// Split string by regex
inline std::vector<std::string> split(const std::string& input, const std::string& regex) {
    // passing -1 as the submatch index parameter performs splitting
    std::regex re(regex);
    std::sregex_token_iterator
        first{ input.begin(), input.end(), re, -1 },
        last;
    return { first, last };
}
#endif