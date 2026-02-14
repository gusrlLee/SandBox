#ifndef SANDBOX_RENDERER_CUDA_HEADER
#define SANDBOX_RENDERER_CUDA_HEADER

#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "kernels.cuh"

__global__ void initRandState(int width, int height, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    int pIdx = j * width + i;
    curand_init(1984 + pIdx, 0, 0, &rand_state[pIdx]);
}

// wavefront path tracing
void render(
    float3 *fb,
    int width, int height, int spp, int maxDepth,
    Camera cam,
    Triangle *triangles, int numTriangles,
    Material *materials, int numMaterials,
    RayWorkItem *dQueueCurrent,
    RayWorkItem *dQueueNext,
    HitQueue *dHitQueue,
    PixelState *dPixelStates,
    int *dNumRays,
    int *dNextNumRays,
    curandState *randState)
{
    int tx = 8, ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    initRandState<<<blocks, threads>>>(width, height, randState);
    cudaDeviceSynchronize();

    for (int s = 0; s < spp; s++)
    {
        // [Step 1] Generate Rays (Primary Rays)
        // 초기화: d_numRays를 width * height로 설정하고, 모든 픽셀에 대해 레이 생성
        generateRays<<<blocks, threads>>>(
            width, height, cam, s, 
            dQueueCurrent, dPixelStates, dNumRays, randState
        );
        cudaDeviceSynchronize(); // Generate 완료 대기

        int h_numRays = 0; // 호스트에서 현재 레이 개수 추적용

        for (int depth = 0; depth < maxDepth; depth++)
        {
            // Device에 있는 레이 개수를 Host로 가져옴
            cudaMemcpy(&h_numRays, dNumRays, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (h_numRays == 0) break; // 처리할 레이가 없으면 조기 종료

            // 다음 단계 카운터(Atomic용) 0으로 초기화
            cudaMemset(dNextNumRays, 0, sizeof(int));

            // Grid 크기 계산 (1D Grid)
            int numBlocks = (h_numRays + 255) / 256;

            // [Step 2] Extend (Intersection Test)
            // 현재 큐(dQueueCurrent)에 있는 h_numRays 만큼만 수행
            extend<<<numBlocks, 256>>>(
                h_numRays, 
                dQueueCurrent, dHitQueue, 
                triangles, numTriangles
            );
            cudaDeviceSynchronize(); // 디버깅 시 필요할 수 있음

            // [Step 3] Shade & Enqueue (Logic)
            // 교차 결과를 바탕으로 쉐이딩하고, 살아남은 레이를 d_queueNext에 넣음
            shadeAndEnqueue<<<numBlocks, 256>>>(
                h_numRays, 
                dNextNumRays, 
                dQueueCurrent, dHitQueue, 
                dQueueNext, 
                dPixelStates, fb, 
                triangles, materials, randState, 
                maxDepth
            );
            cudaDeviceSynchronize(); 

            // [Step 4] Ping-Pong (Swap Queues)
            // 큐 포인터 교체 (다음 루프에서는 next가 current가 됨)
            std::swap(dQueueCurrent, dQueueNext);
            
            // 레이 개수 업데이트: d_nextNumRays(계산된 값) -> dNumRays(다음 루프 입력)
            cudaMemcpy(dNumRays, dNextNumRays, sizeof(int), cudaMemcpyDeviceToDevice);
        }

        // 진행 상황 표시 (10번째 샘플마다)
        if ((s + 1) % 10 == 0) 
            std::cout << "Sample: " << (s + 1) << " / " << spp << "\r" << std::flush;
    }

    std::cout << std::endl << "[Info] Rendering Done." << std::endl;
}

#endif