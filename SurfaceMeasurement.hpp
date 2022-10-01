#ifndef _SURFACEMEASUREMENT_HPP
#define _SURFACEMEASUREMENT_HPP

#include "Utils.h"
#include "GPUProgramLoader.hpp"

class SurfaceMeasurement {
    public:
        SurfaceMeasurement(cl_kernel vertexMapKernel, cl_kernel normaMaplKernel, cl_command_queue queue) {
            m_normalMapKernel = normaMaplKernel;
            m_vertexMapKernel = vertexMapKernel;
            m_queue = queue;
        }

		// GPU Kernel Function Call ** VertexMapKernel **
		// Calculate the Vertex map of given frame in GPU
		// @in depth depth frame in GPUBuffer
		// @in intrinsicInv inverse of camera intrinsic
		// @out vertexMap
		void SurfaceMeasurement::calculateVertexMap(GPUBuffer depth, Eigen::Matrix3f intrinsicInv, GPUBuffer vertexMap) {

			cl_mem inDepthMap = depth.getDeviceBuffer();

			cl_mem outVertexMap = vertexMap.getDeviceBuffer();

			GPUMatrix3f inIntrinsicInv = toGPUMatrix3f(intrinsicInv);

			// Add parameters
			cl_int err;
			cl_int counter = 0;

			// Kernel Parameters In 0, 1
			err = clSetKernelArg(m_vertexMapKernel, counter, sizeof(cl_mem), (void*)&inDepthMap); // 0
			assert(err == CL_SUCCESS);

			err = clSetKernelArg(m_vertexMapKernel, ++counter, sizeof(GPUMatrix3f), (void*)&inIntrinsicInv); // 1
			assert(err == CL_SUCCESS);

			// Kernel Parameters Out 2
			err = clSetKernelArg(m_vertexMapKernel, ++counter, sizeof(cl_mem), (void*)&outVertexMap); // 2
			assert(err == CL_SUCCESS);

			// Add kernel to execution queue
			err = clEnqueueNDRangeKernel(m_queue, m_vertexMapKernel, 2, NULL, getWorkSize(), NULL, 0, NULL, NULL);
			assert(err == CL_SUCCESS);

			float* vertexHostBuffer = vertexMap.getHostBuffer();
			int vertexBufferSize = vertexMap.getBufferSize();
			// Get it back into  Memory Host
			err = clEnqueueWriteBuffer(m_queue, outVertexMap, CL_TRUE, 0, sizeof(float) * vertexBufferSize, vertexHostBuffer, 0, NULL, NULL);
			assert(err == CL_SUCCESS);
		}

		// GPU Kernel Function Call ** NormalMapKernel **
		// Calculate the Normal map of given VertexMap in GPU
		// @in vertexMap 
		// @out normalMap
        void calculateNormalMap(GPUBuffer vertexMap, GPUBuffer normalMap) {
			cl_mem vertexDeviceBuffer = vertexMap.getDeviceBuffer();

			cl_mem normalDeviceBuffer = normalMap.getDeviceBuffer();

			cl_int err;
			cl_int counter = 0;

			// Kernel Parameters In 0
			err = clSetKernelArg(m_normalMapKernel, counter, sizeof(cl_mem), (void*)&vertexDeviceBuffer);
			assert(err == CL_SUCCESS);

			// Kernel Parameters Out 1
			err = clSetKernelArg(m_normalMapKernel, ++counter, sizeof(cl_mem), (void*)&normalDeviceBuffer);
			assert(err == CL_SUCCESS);

			// Add kernel to execution queue
			err = clEnqueueNDRangeKernel(m_queue, m_normalMapKernel, 2, NULL, getWorkSize(), NULL, 0, NULL, NULL);
			assert(err == CL_SUCCESS);

			int normalBufferSize = normalMap.getBufferSize();
			float* normalHostBuffer = normalMap.getHostBuffer();
			// Get it back into Memory Host
			err = clEnqueueWriteBuffer(m_queue, vertexDeviceBuffer, CL_TRUE, 0, sizeof(float) * normalBufferSize, normalHostBuffer, 0, NULL, NULL);
			assert(err == CL_SUCCESS);
		}

    private:
        cl_kernel m_vertexMapKernel;
        cl_kernel m_normalMapKernel;
        cl_command_queue m_queue;
};

#endif // !_SURFACEMEASUREMENT_HPP