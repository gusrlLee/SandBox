#ifndef SANDBOX_KERNELS_CUDA_HEADER
#define SANDBOX_KERNELS_CUDA_HEADER

#include "common.cuh"
#include "rtcore.cuh"

__device__ float3 randomInUnitSphere(curandState *localRandState)
{
    float3 p;
    do
    {
        // -1.0 ~ 1.0 사이의 랜덤 값 추출
        float r1 = curand_uniform(localRandState);
        float r2 = curand_uniform(localRandState);
        float r3 = curand_uniform(localRandState);
        p = 2.0f * make_float3(r1, r2, r3) - make_float3(1.0f, 1.0f, 1.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

// Wavefront path tracing
__global__ void generateRays(
    int width, int height,
    Camera cam, int spp_iter,
    RayWorkItem *rayQ,
    PixelState *stateQ,
    int *queueSize, curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height))
        return;

    int pIdx = j * width + i;
    curandState localRand = randState[pIdx];

    float u = (float(i) + curand_uniform(&localRand)) / float(width);
    float v = (float(j) + curand_uniform(&localRand)) / float(height);

    float3 dir = normalize(cam.lowerLeftCorner + u * cam.horizontal + v * cam.vertical - cam.origin);
    rayQ[pIdx].origin = cam.origin;
    rayQ[pIdx].direction = dir;
    rayQ[pIdx].pixelIndex = pIdx;

    stateQ[pIdx].throughput = make_float3(1.0f);
    stateQ[pIdx].accumulatedColor = make_float3(0.0f);
    stateQ[pIdx].depth = 0;

    randState[pIdx] = localRand;

    if (pIdx == 0)
        *queueSize = width * height;
}

__global__ void extend(
    int queueSize,
    RayWorkItem *rayQ, HitQueue *hitQ,
    Triangle *triangles, int numTriangles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= queueSize)
        return;

    RayWorkItem r = rayQ[idx];
    Ray ray(r.origin, r.direction);

    RayPayload payload;
    float closest = 1e20f;
    int hitIdx = -1;

    RayPayload tmp;
    for (int i = 0; i < numTriangles; i++)
    {
        if (intersectTriangle(triangles[i], ray, 0.001f, closest, tmp))
        {
            closest = tmp.t;
            hitIdx = i;
            payload = tmp;
        }
    }

    hitQ[idx].triIdx = hitIdx;
    hitQ[idx].t = closest;
}

__global__ void shadeAndEnqueue(
    int numCurrentRays,
    int *d_nextQueueCount,
    const RayWorkItem *currentQ,
    const HitQueue *hitQ,
    RayWorkItem *nextQ,
    PixelState *pixelStates,
    float3 *frameBuffer,
    Triangle *triangles,
    Material *materials,
    curandState *randState,
    int maxDepth)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numCurrentRays) return;

    RayWorkItem rItem = currentQ[idx];
    HitQueue hit = hitQ[idx];
    int pIdx = rItem.pixelIndex;
    PixelState &state = pixelStates[pIdx];

    if (hit.triIdx == -1) {
        return;
    }

    Triangle tri = triangles[hit.triIdx];
    Material mat = materials[tri.matId];

    float3 emission = mat.emission; 
    
    if (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f) {
        state.accumulatedColor += state.throughput * emission;
        frameBuffer[pIdx] += state.accumulatedColor;
        return;
    }

    if (state.depth >= maxDepth) {
        frameBuffer[pIdx] += state.accumulatedColor;
        return;
    }

    float3 hitPoint = rItem.origin + rItem.direction * hit.t;
    float3 normal = tri.normal();
    float3 outward_normal = (dot(rItem.direction, normal) < 0) ? normal : -normal;

    float3 albedo = mat.albedo; 

    curandState localRand = randState[pIdx];
    float3 randomVec = randomInUnitSphere(&localRand); // 구현해두신 함수 사용
    float3 target = outward_normal + randomVec;
    
    if (dot(target, target) < 1e-3f) target = outward_normal;
    
    float3 nextDir = normalize(target);

    state.throughput *= albedo;

    state.depth++;
    randState[pIdx] = localRand; // 랜덤 상태 저장

    int slot = atomicAdd(d_nextQueueCount, 1);
    RayWorkItem nextItem;
    nextItem.pixelIndex = pIdx;
    nextItem.origin = hitPoint + outward_normal * 1e-4f; // Self-intersection 방지
    nextItem.direction = nextDir;
    nextQ[slot] = nextItem;
}

#endif