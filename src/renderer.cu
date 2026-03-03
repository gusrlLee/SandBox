#include "renderer.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

Renderer::Renderer()
{
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
}

Renderer::~Renderer()
{

}