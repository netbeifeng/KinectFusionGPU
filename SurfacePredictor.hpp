#ifndef _SURFACEPREDICTOR_HPP
#define _SURFACEPREDICTOR_HPP

#include "Utils.h"
#include "GPUProgramLoader.hpp"

class SurfacePredictor {
    public:
        SurfacePredictor(cl_context ctx, cl_kernel kernel, cl_command_queue queue) {
            m_kernel = kernel;
            m_ctx = ctx;
            m_queue = queue;
        }

        // GPU Kernel Function Call ** RayCasterKernel **
        // @in tsdfValuesBuffer values saved in globalTSDF
        // @in tsdfColoursBuffer colours value saved in globalTSDF
        // @in transform pose of camera
        void raycast(GPUBuffer tsdfValuesBuffer, GPUBuffer tsdfColoursBuffer, Eigen::Matrix4f transform) {
            cl_int err;
            cl_int counter = 0;

            GPUMatrix3f inIntrinsicInv = toGPUMatrix3f(Eigen::Matrix3f(getIntrinsic().inverse()));
            GPUMatrix4f inTransformation = toGPUMatrix4f(transform);
            GPUMatrix4f inTransfornationInv = toGPUMatrix4f(Eigen::Matrix4f(transform.inverse()));

            // Kernel In 0, 1, 2 Parameters
            err = clSetKernelArg(m_kernel, counter, sizeof(GPUMatrix3f), (void*)&inIntrinsicInv); // 0
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_kernel, ++counter, sizeof(GPUMatrix4f), (void*)&inTransformation); // 1
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_kernel, ++counter, sizeof(GPUMatrix4f), (void*)&inTransfornationInv); // 2
            assert(err == CL_SUCCESS);

            // Kernel In 3, 4 Parameters
            cl_mem inTSDFColoursMap = tsdfColoursBuffer.getDeviceBuffer();
            cl_mem inTSDFValuesMap = tsdfValuesBuffer.getDeviceBuffer();
            err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&inTSDFValuesMap); // 3
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&inTSDFColoursMap); // 4
            assert(err == CL_SUCCESS);

            // Kernel Out 5, 6 Parameters
            cl_mem outDepthMap = m_depthMap.getDeviceBuffer();
            cl_mem outColourMap = m_colourMap.getDeviceBuffer();
            err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&outDepthMap);
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&outColourMap);
            assert(err == CL_SUCCESS);

            // Push Kernel to OpenCL execution Queue
            err = clEnqueueNDRangeKernel(m_queue, m_kernel, 2, NULL, getWorkSize(), NULL, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            int depthMapBufferSize = m_depthMap.getBufferSize();
            float* depthMapHostBuffer = m_depthMap.getHostBuffer();

            int colourMapBufferSize = m_colourMap.getBufferSize();
            float* colourMapHostBuffer = m_colourMap.getHostBuffer();

            // Get the calculated result back to host memory
            // Depth Frame retrival
            err = clEnqueueReadBuffer(m_queue, outDepthMap, CL_TRUE, 0, sizeof(float) * depthMapBufferSize, depthMapHostBuffer, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            // RGB Frame retrival
            err = clEnqueueReadBuffer(m_queue, outColourMap, CL_TRUE, 0, sizeof(float) * colourMapBufferSize, colourMapHostBuffer, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        }
        
        void setColourMapBuffer(GPUBuffer& buffer) {
            m_colourMap = buffer;
        }

        void setDepthMapBuffer(GPUBuffer& buffer) {
            m_depthMap = buffer;
        }

        // Show RayCast frame in OpenCV
        void showRayCastFrame() {
            cv::Mat rgbFrame = convertToMat(m_colourMap, RGB_CHANNEL);
            cv::Mat depthFrame = convertToMat(m_depthMap, DEPTH_CHANNEL);
            cv::imshow("RayCast_RGB", rgbFrame);
            cv::imshow("RayCast_Depth", depthFrame);
            cv::waitKey();
        }

    private:
        cl_command_queue m_queue;
        cl_kernel m_kernel;
        cl_context m_ctx;

        GPUBuffer m_colourMap;
        GPUBuffer m_depthMap;

        // Convert float* to OpenCV Mat
        cv::Mat convertToMat(GPUBuffer b, int channel) {
            float max, min;
            float* data = b.getHostBuffer();
            calcMinMax(data, channel, min, max);
            cv::Mat mData;
            if (channel == 1) {
                mData = cv::Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_32FC1);
            }
            else {
                mData = cv::Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_32FC3);
            }

            fillToMat(mData, data);
            if (channel == 1) {
                mData = (mData - min) / (max - min);
            }
            return mData;
        }

        // get min and max of float*
        inline void calcMinMax(const float* data, int channel, float& minVal, float& maxVal)
        {
            maxVal = NAN;
            minVal = NAN;
            for (int i = 0; i < FRAME_WIDTH * FRAME_HEIGHT * channel; i++)
            {
                if (!isnan(data[i])) maxVal = isnan(maxVal) ? data[i] : std::max(maxVal, data[i]);
                if (!isnan(data[i])) minVal = isnan(minVal) ? data[i] : std::min(minVal, data[i]);
            }
        }

        // Fill cv::Mat with float*
        inline void fillToMat(cv::Mat& dst, const float* src) {
            int channel = dst.channels();
            float* data = (float*)dst.data;
            for (int c = 0; c < channel; c++)
            {
                for (int x = 0; x < FRAME_WIDTH; ++x)
                {
                    for (int y = 0; y < FRAME_HEIGHT; ++y)
                    {
                        data[channel * (x + FRAME_WIDTH * y) + channel - 1 - c] = \
                            static_cast<float>(src[(x + FRAME_WIDTH * y) + FRAME_HEIGHT * FRAME_WIDTH * c]);
                    }
                }
            }
        }
};

#endif