#include "render/TestKernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void drawGradientKernel(float4* pbo, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 화면 밖을 벗어나는 스레드 종료
    if (x >= width || y >= height) return;

    // 1차원 배열 인덱스 계산
    int index = y * width + x;

    // UV 좌표를 색상(R, G)으로 변환
    float r = (float)x / width;
    float g = (float)y / height;
    float b = 0.2f; // 약간의 푸른빛 추가

    // PBO 메모리에 직접 RGBA 쓰기
    pbo[index] = make_float4(r, g, b, 1.0f);
}


void launchTestKernel(void* d_pbo, int width, int height) 
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    drawGradientKernel <<<grid, block >>> ((float4*)d_pbo, width, height);
    cudaDeviceSynchronize();
}

