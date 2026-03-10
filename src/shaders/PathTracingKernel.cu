#include <cuda_runtime.h>
#include <optix_device.h>

#include "shaders/Common.h"

extern "C" {
    __constant__ LaunchParams params;
}

// 1. Ray Generation (카메라에서 광선을 쏘는 역할)
extern "C" __global__ void __raygen__rg()
{
    // 현재 실행 중인 픽셀의 2D 인덱스 (x, y)
    const uint3 idx = optixGetLaunchIndex();
    // 전체 화면 크기 (width, height)
    const uint3 dim = optixGetLaunchDimensions();

    // 간단한 UV 그라데이션 색상 계산
    float r = (float)idx.x / (float)dim.x;
    float g = (float)idx.y / (float)dim.y;
    float b = 0.2f;

    // uchar4 (0~255) 형태로 변환
    uchar4 color = make_uchar4(
        (unsigned char)(r * 255.0f),
        (unsigned char)(g * 255.0f),
        (unsigned char)(b * 255.0f),
        255
    );

    // 1D 배열 형태인 colorBuffer에 현재 픽셀의 색상 기록
    uint32_t pixelIndex = idx.y * params.width + idx.x;
    params.colorBuffer[pixelIndex] = color;
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