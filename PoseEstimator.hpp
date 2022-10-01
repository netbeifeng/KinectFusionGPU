#ifndef _POSEESTIMATOR_HPP
#define _POSEESTIMATOR_HPP

#include "Utils.h"

#include <pcl/registration/gicp.h>
#include <pcl/registration/gicp6d.h>

class PoseEstimator {
    public:
        PoseEstimator(cl_kernel dataAlignKernel, cl_command_queue queue) {
            m_dataAsscKernel = dataAlignKernel;
            m_maxIterNum = ICP_MAX_ITER;
            m_queue = queue;
        }

        // CPU based GICP Color
        // Slow 3 secs 
        std::pair<Eigen::Matrix4f, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> computeOnCPU(
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr current,
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr next) {

                pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> gicp;
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr alignOutput(new pcl::PointCloud<pcl::PointXYZRGBA>);
                gicp.setInputSource(current);
                gicp.setInputTarget(next);

                gicp.align(*alignOutput);
                if (m_maxIterNum > 0) {
                    gicp.setMaximumIterations(m_maxIterNum);
                }
                return std::make_pair(gicp.getFinalTransformation(), alignOutput);
        }

        // GPU Kernel Function Call ** DataAssociationKernel **
        // Make sure all points are well associated before using for ICP
        // @in rayCastVertexMap last raycasted vertexmap
        // @in rayCastNormalMap last raycasted normalmap
        // @in currVertexMap curr calculated vertexmap
        // @in currNormalMap curr calculated normalmap
        // @in transInv inverse of pose
        // @out corr corrsponding pixels
        void associateData(GPUBuffer rayCastVertexMap, GPUBuffer rayCastNormalMap,
            GPUBuffer currVertexMap, GPUBuffer currNormalMap, Eigen::Matrix4f transInv, GPUBuffer corr) {

            cl_int err;
            cl_int counter = 0;

            cl_mem inLastVertexMap = rayCastVertexMap.getDeviceBuffer();
            cl_mem inLastNormalMap = rayCastNormalMap.getDeviceBuffer();

            cl_mem inCurrVertexMap = rayCastVertexMap.getDeviceBuffer();
            cl_mem inCurrNormalMap = rayCastNormalMap.getDeviceBuffer();

            cl_mem outCorrsp = corr.getDeviceBuffer();

            GPUMatrix4f transformationInv = toGPUMatrix4f(transInv);

            // Kernel In 0, 1, 2, 3 , 4 
            err = clSetKernelArg(m_dataAsscKernel, counter, sizeof(GPUMatrix4f), (void*)&transformationInv); // 0
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_dataAsscKernel, ++counter, sizeof(cl_mem), (void*)&inLastVertexMap);  // 1
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_dataAsscKernel, ++counter, sizeof(cl_mem), (void*)&inLastNormalMap); // 2
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_dataAsscKernel, ++counter, sizeof(cl_mem), (void*)&inCurrVertexMap); // 3
            assert(err == CL_SUCCESS);
            err = clSetKernelArg(m_dataAsscKernel, ++counter, sizeof(cl_mem), (void*)&inCurrNormalMap); // 4
            assert(err == CL_SUCCESS);

            // Kernel Out 5 
            err = clSetKernelArg(m_dataAsscKernel, ++counter, sizeof(cl_mem), (void*)&outCorrsp); // 5
            assert(err == CL_SUCCESS);

            // Push Kernel to OpenCL execution Queue
            err = clEnqueueNDRangeKernel(m_queue, m_dataAsscKernel, 2, NULL, getWorkSize(), NULL, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            // Get the calculated result back to host memory
            err = clEnqueueReadBuffer(m_queue, corr.getDeviceBuffer(), CL_TRUE, 0, sizeof(float) * corr.getBufferSize(), corr.getHostBuffer(), 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        }
    private:
        int m_maxIterNum;
        cl_kernel m_dataAsscKernel;
        cl_command_queue m_queue;
};

#endif