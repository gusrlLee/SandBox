#include <iostream>

#include "renderer.cuh"
#include "window.cuh"

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
    std::shared_ptr<Window> window = std::make_shared<Window>("Check", 800, 800);
    std::shared_ptr<Renderer> renderer = std::make_shared<Renderer>();

    while(!window->shouldClose())
    {
        cudaSurfaceObject_t frame;
        window->beginFrame(frame);

        launchCascadeKernel(frame, 800, 800);

        window->endFrame(frame);
        window->swapBuffer();
    }


}