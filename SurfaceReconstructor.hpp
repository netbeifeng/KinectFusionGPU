#ifndef _SURFACERECONSTRUCTOR_HPP
#define _SURFACERECONSTRUCTOR_HPP

#include "GPUProgramLoader.hpp"

class SurfaceReconstructor {
	public:
		SurfaceReconstructor(cl_context ctx, cl_kernel kernel, cl_command_queue queue) {
			m_kernel = kernel;
			m_queue = queue;
			m_ctx = ctx;
		}

		void SurfaceReconstructor::setValuesBuffer(GPUBuffer& buffer) {
			m_valuesBuffer = buffer;
		}

		void SurfaceReconstructor::setWeightsBuffer(GPUBuffer& buffer) {
			m_weightsBuffer = buffer;
		}

		void SurfaceReconstructor::setColoursBuffer(GPUBuffer& buffer) {
			m_coloursBuffer = buffer;
		}

		// GPU Kernel Function Call ** TSDFIntegrationKernel **
		// @in depth camera depth frame
		// @in rgb camera colour frame
		// @in transformInv inverse of pose
		void executeIntegration(GPUBuffer depth, GPUBuffer rgb, Eigen::Matrix4f _transformationInv) {

			cl_int err;
			cl_int counter = 0;

			cl_mem outTSDFValues = m_valuesBuffer.getDeviceBuffer();
			cl_mem outTSDFWeights = m_weightsBuffer.getDeviceBuffer();
			cl_mem outTSDFColours = m_coloursBuffer.getDeviceBuffer();
			cl_mem inDepthMap = depth.getDeviceBuffer();
			cl_mem inColourMap = rgb.getDeviceBuffer();

			GPUMatrix4f transformationInv = toGPUMatrix4f(_transformationInv);

			// Kernel Parameters 0, 1, 2
			err = clSetKernelArg(m_kernel, counter, sizeof(GPUMatrix4f), (void*)&transformationInv); // 0
			assert(err == CL_SUCCESS);
			err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&inDepthMap); // 1
			assert(err == CL_SUCCESS);
			err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&inColourMap); // 2
			assert(err == CL_SUCCESS);

			// Kernel Parameters Out 3, 4, 5
			err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&outTSDFValues); // 3
			assert(err == CL_SUCCESS);
			err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&outTSDFColours); // 4
			assert(err == CL_SUCCESS);
			err = clSetKernelArg(m_kernel, ++counter, sizeof(cl_mem), (void*)&outTSDFWeights); // 5
			assert(err == CL_SUCCESS);

			// Add kernel to execution queue
			size_t globalSize[3] = { VOLUME_RES, VOLUME_RES, VOLUME_RES };
			err = clEnqueueNDRangeKernel(m_queue, m_kernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
			assert(err == CL_SUCCESS);

			// No need to download the data to memory host, just let it stay in device and for raycasting
			// Save time!
			// If really need to do so => getResultToHost()
		}

		void getResultToHost() {
			cl_int err;

			int valuesBufferSize = m_valuesBuffer.getBufferSize();
			cl_mem& valuesDeviceBuffer = m_valuesBuffer.getDeviceBuffer();
			float* valuesHostBuffer = m_valuesBuffer.getHostBuffer();

			err = clEnqueueReadBuffer(m_queue, valuesDeviceBuffer, CL_TRUE, 0, sizeof(float) * valuesBufferSize, valuesHostBuffer, 0, NULL, NULL);
			assert(err == CL_SUCCESS);

			// At least no requirement to get Weights buffer also unless debugging
			//int weightsBufferSize = m_weightsBuffer.getBufferSize();
			//cl_mem& weightsDeviceBuffer = m_weightsBuffer.getDeviceBuffer();
			//float* weightsHostBuffer = m_weightsBuffer.getHostBuffer();

			//err = clEnqueueReadBuffer(m_queue, weightsDeviceBuffer, CL_TRUE, 0, sizeof(float) * weightsBufferSize, weightsHostBuffer, 0, NULL, NULL);
			//assert(err == CL_SUCCESS);

			int coloursBufferSize = m_coloursBuffer.getBufferSize();
			cl_mem& coloursDeviceBuffer = m_coloursBuffer.getDeviceBuffer();
			float* coloursHostBuffer = m_coloursBuffer.getHostBuffer();

			err = clEnqueueReadBuffer(m_queue, coloursDeviceBuffer, CL_TRUE, 0, sizeof(float) * coloursBufferSize, coloursHostBuffer, 0, NULL, NULL);
			assert(err == CL_SUCCESS);
		}
	private:
		cl_kernel m_kernel;
		cl_context m_ctx;
		cl_command_queue m_queue;
		GPUBuffer m_valuesBuffer;
		GPUBuffer m_weightsBuffer;
		GPUBuffer m_coloursBuffer;
};

#endif // !_SURFACERECONSTRUCTOR_HPP
