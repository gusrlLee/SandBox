#include <cuda_runtime.h>
#include <optix_device.h>

#include "shaders/Common.h"

extern "C" {
    __constant__ LaunchParams params;
}

// 1. Ray Generation (카메라에서 광선을 쏘는 역할)
extern "C" __global__ void __raygen__rg()
{
    // 현재 스레드의 픽셀 좌표 (x, y)
    uint3 idx = optixGetLaunchIndex();

    if (idx.x >= params.width || idx.y >= params.height) return;

    // 1차원 인덱스 계산
    int pbo_idx = idx.y * params.width + idx.x;

    // 테스트용 색상 (OptiX가 정상 작동하는지 확인하기 위한 보라빛 그라데이션)
    float r = (float)idx.x / params.width;
    float g = 0.3f;
    float b = (float)idx.y / params.height;

    params.pbo[pbo_idx] = make_float4(r, g, b, 1.0f);
}

// 2. Miss (광선이 아무것도 맞추지 못했을 때 배경색 처리)
extern "C" __global__ void __miss__ms() 
{
    // 지금은 비워둡니다.
}

// 3. Closest Hit (광선이 물체와 가장 먼저 교차했을 때 재질/빛 계산)
extern "C" __global__ void __closesthit__ch() 
{
    // 지금은 비워둡니다.
}