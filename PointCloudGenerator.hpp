#ifndef _POINTCLOUDGENERATOR_HPP
#define _POINTCLOUDGENERATOR_HPP

#include "Utils.h"
#include "GPUProgramLoader.hpp"

class PointCloudGenerator {
    public:
        PointCloudGenerator(Eigen::Matrix3f intrinsic) {
            m_intrinsic = intrinsic;
        };

        // Get Data from GPU
        pcl::PointCloud<pcl::PointXYZRGBA> processFrame(GPUBuffer vertexMapBuffer, GPUBuffer rgbFrameBuffer) {
            //pcl::PointCloud<pcl::PointXYZ> processFrame(OpenCLBuffer<cl_float> vertexMapBuffer, OpenCLBuffer<cl_float> rgbFrameBuffer) {
            const cl_float* vertexMap = vertexMapBuffer.getHostBuffer();
            const cl_float* rgbFrame = rgbFrameBuffer.getHostBuffer();

            pcl::PointCloud<pcl::PointXYZRGBA> generatedPointCloud;
            for (cl_uint i = 0; i < FRAME_WIDTH; ++i) {
                for (cl_uint j = 0; j < FRAME_HEIGHT; ++j) {
                    // Index method 
                    // In OpenCV flatten one channel matrix 0,0 -> 0,1
                    // In OpenCV three channels matrix (0,0)[0] (0,0)[1] (0,0)[2] -> (0,1)...
                    // Processed re-index by OpenCL => (0,0) (0,0)+W*H (0,0)+2*W*H => no more continous
                    cl_float x = vertexMap[(i + FRAME_WIDTH * j)];
                    cl_float y = vertexMap[(i + FRAME_WIDTH * j) + FRAME_WIDTH * FRAME_HEIGHT];
                    cl_float z = vertexMap[(i + FRAME_WIDTH * j) + FRAME_WIDTH * FRAME_HEIGHT * 2];
                    if (isnan(x) || isnan(y) || isnan(z)) { continue; }
                    cl_uint b = rgbFrame[(i + FRAME_WIDTH * j)] * 255.0;
                    cl_uint g = rgbFrame[(i + FRAME_WIDTH * j) + FRAME_WIDTH * FRAME_HEIGHT] * 255.0;
                    cl_uint r = rgbFrame[(i + FRAME_WIDTH * j) + FRAME_WIDTH * FRAME_HEIGHT * 2] * 255.0;

                    pcl::PointXYZRGBA point = pcl::PointXYZRGBA(x, y, z, r, g, b, 255);
                    generatedPointCloud.push_back(point);
                }
            }

            return generatedPointCloud;
        }

        // CPU
        pcl::PointCloud<pcl::PointXYZRGBA> processFrame(Frame frame) {
            pcl::PointCloud<pcl::PointXYZRGBA> generatedPointCloud;
            cv::Mat depthFrame = frame.getDepthFrame();

            cv::Mat rgbFrame = frame.getRGBFrame();

            // i for which row, j for which column
            for (int i = 0; i < FRAME_HEIGHT; i++) {
                for (int j = 0; j < FRAME_WIDTH; j++) {
                    cv::Vec3f pixelColor = rgbFrame.at<cv::Vec3f>(i, j);
                    
                    // OpenCV arrange in BGR
                    int r = pixelColor[2] * 255.0; 
                    int g = pixelColor[1] * 255.0;
                    int b = pixelColor[0] * 255.0;

                    float z = depthFrame.at<float>(i, j);
                    float x = (j - m_intrinsic(0, 2)) * z / m_intrinsic(0, 0);
                    float y = (i - m_intrinsic(1, 2)) * z / m_intrinsic(1, 1);

                    pcl::PointXYZRGBA point = pcl::PointXYZRGBA(x, y, z, r, g, b, 255);
                    generatedPointCloud.push_back(point);
                }
            }
            return generatedPointCloud;
        }
    private:
        Eigen::Matrix3f m_intrinsic;
};
#endif