#include <iostream>

#include "window.cuh"

__global__ void cascadeTestKernel(cudaSurfaceObject_t surface, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // 테스트용: 화면 좌표를 기반으로 색상 생성
        float u = (float)x / (float)width;
        float v = (float)y / (float)height;
        float4 color = make_float4(u, v, 0.5f, 1.0f); // R, G, B, A

        // surf2Dwrite는 x 좌표를 바이트 단위로 계산해야 합니다.
        surf2Dwrite(color, surface, x * sizeof(float4), y);
    }
}

void launchCascadeKernel(cudaSurfaceObject_t surface, int width, int height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cascadeTestKernel<<<gridSize, blockSize>>>(surface, width, height);

    // 에러 체크 및 동기화 (디버깅에 유용합니다)
    cudaDeviceSynchronize();
}

int main()
{
    std::shared_ptr<Window> window = std::make_shared<Window>("Check", 800, 800);
    while(!window->shouldClose())
    {
        cudaSurfaceObject_t frame;
        window->beginFrame(frame);

        launchCascadeKernel(frame, 800, 800);

        window->endFrame(frame);
        window->swapBuffer();
    }


}