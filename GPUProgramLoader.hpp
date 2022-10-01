#ifndef _GPUPROGRAMLOADER_HPP
#define _GPUPROGRAMLOADER_HPP

#include <map>
#include <fstream>

#include <CL/cl.h>
#include "Utils.h"

class GPUProgramLoader {
    public:
        GPUProgramLoader() {

            std::vector<std::pair<std::string, std::string>> programKernels;

            programKernels.push_back(std::make_pair("opencl/SurfaceMeasurement.cl", "VertexMapKernel|NormalMapKernel"));
            programKernels.push_back(std::make_pair("opencl/SurfaceReconstructor.cl", "TSDFIntegrationKernel"));
            programKernels.push_back(std::make_pair("opencl/SurfacePredictor.cl", "RayCasterKernel"));
            programKernels.push_back(std::make_pair("opencl/PoseEstimator.cl", "DataAssociationKernel"));

#ifdef _WIN64 
            _putenv_s("CUDA_CACHE_DISABLE", "1");
#elif _WIN32 
            _putenv_s("CUDA_CACHE_DISABLE", "1");
#elif __linux__ 
            setenv("CUDA_CACHE_DISABLE", "1", 1);
#endif
            cl_int err;
            // Get avaliable platform
            cl_uint platformCount = 0;
            clGetPlatformIDs(0, nullptr, &platformCount);
            std::vector<cl_platform_id> platforms(platformCount);
            clGetPlatformIDs(platformCount, platforms.data(), nullptr);

            if (platformCount == 0)
            {
                throw std::invalid_argument("GPUProgramLoader >> No available platform found ...");
            }

            // Get availiable device
            cl_uint deviceCount = 0;
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
            std::vector<cl_device_id> devices(deviceCount);
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

            // Create OpenCL context
            const cl_context_properties contextProps[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platforms[0]), 0, 0 };
            m_ctx = clCreateContext(contextProps, deviceCount, devices.data(), nullptr, nullptr, &err);
            assert(err == CL_SUCCESS);

            // Create OpenCL Queue
            m_queue = clCreateCommandQueue(m_ctx, devices[0], 0, &err);
            assert(err == CL_SUCCESS);

            int counter = 0;
            // Build program files
            for (std::pair<std::string, std::string> pair : programKernels) {
                cl_program program = createProgramFromSource(m_ctx, pair.first, devices);
                // Create kernels of each program, different kernel divided by pipe symbol
                for (auto s : split(pair.second, "\\\|")) {
                    std::cout << "GPUProgramLoader >> Kernel " << s << std::endl;
                    m_kernels[s] = clCreateKernel(program, s.data(), &err);
                    assert(err == CL_SUCCESS);
                }
                counter++;
            }
        }

        cl_kernel getKernelByKey(std::string key) {
            return m_kernels[key];
        }

        cl_context getContext() {
            return m_ctx;
        }

        cl_command_queue getQueue() {
            return m_queue;
        }
        
    private:
        std::map<std::string, cl_kernel> m_kernels;
        cl_command_queue m_queue;
        cl_context m_ctx;
        std::vector<cl_program> m_programs;

        cl_program createProgramFromSource(cl_context context, const std::string fileName, std::vector<cl_device_id> devices)
        {
            cl_int err;
            std::cout << "GPUProgramLoader >> Compiling " << fileName << " ..." << std::endl;
            //Read source file into string
            std::ifstream ifs("../" + fileName);
            std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

            // Prepare parameters for program creation   
            const char* sourceArray[1] = { programSource.data() };
            size_t lengthArray[1] = { programSource.size() };

            // Create program
            cl_program result = clCreateProgramWithSource(context, 1, sourceArray, lengthArray, &err);
            assert(err == CL_SUCCESS);

            // Build program with ../opencl as header source
            err = clBuildProgram(result, 1, &devices[0], "-I ../opencl/", NULL, NULL);

            if (err == CL_BUILD_PROGRAM_FAILURE) {
                // Determine the size of the log
                size_t log_size;
                clGetProgramBuildInfo(result, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

                // Allocate memory for the log
                char* log = (char*)malloc(log_size);

                // Get the log
                clGetProgramBuildInfo(result, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

                // Print the log
                printf("OpenCLManager >> Error Log:\n%s\n", log);
                free(log);
                throw std::logic_error("Error");
            }
            else {
                assert(err == CL_SUCCESS);
                std::cout << "OpenCLManager >> " << fileName << " Successfully Compilied." << std::endl;
            }

            return result;
        }
};

class GPUBuffer {
public:
	GPUBuffer() {}

	GPUBuffer(cl_context ctx, cl_command_queue queue, cv::Mat frame) {
		cv::Size size = frame.size();
		int channels = frame.channels();
		cv::Mat flattenMat = frame.reshape(1, size.height * size.width * channels);
		float* data = (float*)flattenMat.data;

		m_ctx = ctx;
		m_queue = queue;
		m_bufferSize = size.height * size.width * channels;
		m_hostBuffer = new float[m_bufferSize];

		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < size.width; x++) {
				for (int y = 0; y < size.height; y++) {
					m_hostBuffer[(x + size.width * y) + size.width * size.height * c] = data[channels * (x + size.width * y) + channels - 1 - c];
				}
			}
		}

		m_deviceBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * m_bufferSize, NULL, NULL);
		clEnqueueWriteBuffer(m_queue, m_deviceBuffer, CL_TRUE, 0, sizeof(float) * m_bufferSize, m_hostBuffer, 0, NULL, NULL);
	}

	GPUBuffer(cl_context ctx, cl_command_queue queue, float* data, int channels, int width, int height) {
		m_ctx = ctx;
		m_queue = queue;
		m_bufferSize = width * height * channels;
		m_hostBuffer = new float[m_bufferSize];

		for (int c = 0; c < channels; c++) {
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					m_hostBuffer[(x + width * y) + width * height * c] = data[channels * (x + width * y) + channels - 1 - c];
				}
			}
		}

		m_deviceBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * m_bufferSize, NULL, NULL);
		clEnqueueWriteBuffer(m_queue, m_deviceBuffer, CL_TRUE, 0, sizeof(float) * m_bufferSize, m_hostBuffer, 0, NULL, NULL);
	}

	GPUBuffer(cl_context ctx, cl_command_queue queue, int bufferSize) {
		m_ctx = ctx;
		m_queue = queue;
		m_bufferSize = bufferSize;
		m_hostBuffer = new float[m_bufferSize];
		m_deviceBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * m_bufferSize, NULL, NULL);
	}

	GPUBuffer(cl_context ctx, cl_command_queue queue, int bufferSize, float initVal) {
		m_ctx = ctx;
		m_queue = queue;
		m_bufferSize = bufferSize;
		m_hostBuffer = new float[m_bufferSize];
		m_deviceBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * m_bufferSize, NULL, NULL);
		for (int i = 0; i < m_bufferSize; i++) {
			m_hostBuffer[i] = (float)initVal;
		}
		clEnqueueWriteBuffer(m_queue, m_deviceBuffer, CL_TRUE, 0, sizeof(float) * m_bufferSize, m_hostBuffer, 0, NULL, NULL);
	}

	cl_mem& getDeviceBuffer() {
		return m_deviceBuffer;
	}

	float* getHostBuffer() {
		return m_hostBuffer;
	}

	int getBufferSize() {
		return m_bufferSize;
	}

	void clear() {
		free(m_hostBuffer);
		clReleaseMemObject(m_deviceBuffer);
	}

private:
	cl_mem m_deviceBuffer;
	cl_context m_ctx;
	cl_command_queue m_queue;

	int m_bufferSize;
	float* m_hostBuffer;
};

class BufferManager {
public:
	BufferManager(cl_context ctx, cl_command_queue queue) {
		m_ctx = ctx;
		m_queue = queue;
		std::cout << "BufferManager >> Initializing Buffers ..." << std::endl;
		//m_buffers["depth_frame"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT);
		//m_buffers["last_depth_frame"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT);
		//m_buffers["rgb_frame"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT * RGB_CHANNEL);
		//m_buffers["last_rgb_frame"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT * RGB_CHANNEL);

		//m_buffers["vertex_map"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT * RGB_CHANNEL);
		//m_buffers["normal_map"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT * RGB_CHANNEL);

		//m_buffers["raycast_vertex_map"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT * RGB_CHANNEL);
		//m_buffers["raycast_normal_map"] = GPUBuffer(m_ctx, m_queue, FRAME_WIDTH * FRAME_HEIGHT * RGB_CHANNEL);
		std::cout << "BufferManager >> Buffers successfully initialized." << std::endl;
	}

	GPUBuffer getBufferByKey(std::string key) {
		return m_buffers[key];
	}

	void setBufferByKey(std::string key, GPUBuffer buffer) {
		m_buffers[key] = buffer;
	}

private:
	cl_context m_ctx;
	cl_command_queue m_queue;
	std::map<std::string, GPUBuffer> m_buffers;
};
#endif