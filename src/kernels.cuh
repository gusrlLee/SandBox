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

__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const float3 &v, const float3 &n, float ni_over_nt, float3 &refracted)
{
    float3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    return false;
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

    // 1. Miss 처리 (검은 배경)
    if (hit.triIdx == -1) {
        // 빛이 새지 않도록 완전 검은색 처리 (Cornell Box는 닫힌 공간임)
        // frameBuffer[pIdx] += state.accumulatedColor; (필요 없음)
        return;
    }

    Triangle tri = triangles[hit.triIdx];
    Material mat = materials[tri.matId];

    // 2. Emission (빛) 처리 - mtl의 Ke 값을 사용
    // Ke가 (17, 12, 4) 처럼 매우 큽니다.
    float3 emission = mat.emission; 
    
    // 만약 빛나는 물체라면?
    if (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f) {
        // 경로의 가중치(Throughput)만큼 빛을 더함
        state.accumulatedColor += state.throughput * emission;
        
        // [중요] 빛을 만났으면 레이 추적 종료 (Explicit Light Sampling 없으면 여기서 끊는게 깔끔함)
        frameBuffer[pIdx] += state.accumulatedColor;
        return;
    }

    // 3. Max Depth 도달 시 종료
    if (state.depth >= maxDepth) {
        frameBuffer[pIdx] += state.accumulatedColor;
        return;
    }

    // 4. 지오메트리 정보 계산
    float3 hitPoint = rItem.origin + rItem.direction * hit.t;
    float3 normal = tri.normal();
    // Ray가 안쪽에서 맞았는지 바깥쪽에서 맞았는지 판별
    float3 outward_normal = (dot(rItem.direction, normal) < 0) ? normal : -normal;

    // 5. Diffuse (Lambertian) 반사 계산
    // mtl의 Kd (Diffuse Color)를 Albedo로 사용
    float3 albedo = mat.albedo; 

    // Lambertian 반사 방향 (Cosine Weighted Sampling)
    curandState localRand = randState[pIdx];
    float3 randomVec = randomInUnitSphere(&localRand); // 구현해두신 함수 사용
    float3 target = outward_normal + randomVec;
    
    // 타겟이 0이 되는 드문 경우 방지
    if (dot(target, target) < 1e-3f) target = outward_normal;
    
    float3 nextDir = normalize(target);

    // Throughput 업데이트 (Lambertian에서는 albedo만 곱하면 됨)
    // 수학적 유도: (albedo / PI) * (cos_theta) / (pdf: cos_theta / PI) = albedo
    state.throughput *= albedo;

    // 6. 다음 레이 준비 (Next Ray Setup)
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