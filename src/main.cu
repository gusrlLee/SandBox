#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "window.cuh"

#define OPTIX_CHECK(call)                                                                         \
{                                                                                                 \
    OptixResult res = call;                                                                       \
    if (res != OPTIX_SUCCESS)                                                                     \
    {                                                                                             \
        fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
        exit(2);                                                                                  \
    }                                                                                             \
}

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}


__global__ void cascadeTestKernel(cudaSurfaceObject_t surface, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float u = (float)x / (float)width;
        float v = (float)y / (float)height;
        float4 color = make_float4(u, v, 0.5f, 1.0f); // R, G, B, A

        surf2Dwrite(color, surface, x * sizeof(float4), y);
    }
}

void launchCascadeKernel(cudaSurfaceObject_t surface, int width, int height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cascadeTestKernel<<<gridSize, blockSize>>>(surface, width, height);
    cudaDeviceSynchronize();
}

int main()
{
    int width = 1080;
    int height = 720;

    std::shared_ptr<Window> window = std::make_shared<Window>("Path Tracer", width, height);

    cudaFree(0);
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4; // 0 : disable, 4 : info (detail)

#ifdef _DEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
    // context create
    OptixDeviceContext context = nullptr;
    CUcontext cuCtx = 0; // 0 means current CUDA context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    std::cout << "OptiX Context created successfully!" << std::endl;

    // context create

    while(!window->shouldClose())
    {
        cudaSurfaceObject_t frame;
        window->beginFrame(frame);

        launchCascadeKernel(frame, window->getWidth(), window->getHeight());

        window->endFrame(frame);
        window->swapBuffer();
    }

    OPTIX_CHECK(optixDeviceContextDestroy(context));
    return 0;
}