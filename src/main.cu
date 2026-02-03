#include <iostream>
#include <filesystem>

#include "math_helper.cuh"
#include "structs.cuh"

#include "bvh.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void initRandState(int width, int height, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    int pIdx = j * width + i;
    curand_init(1984 + pIdx, 0, 0, &rand_state[pIdx]);
}

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

__device__ float intersectAABB(const Ray &r, const AABB &box)
{
    float3 invDir = make_float3(1.0f / r.dir.x, 1.0f / r.dir.y, 1.0f / r.dir.z);

    float3 t0 = (box.min - r.orig) * invDir;
    float3 t1 = (box.max - r.orig) * invDir;

    float3 tmin = fminf(t0, t1);
    float3 tmax = fmaxf(t0, t1);

    // 3축 중 가장 늦게 들어가는 시점과 가장 빨리 나가는 시점 계산
    float tEnter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float tExit = fminf(fminf(tmax.x, tmax.y), tmax.z);

    // 유효한 충돌인지 확인 (박스 뒤에 있거나, 진입이 진출보다 늦으면 꽝)
    if (tExit >= tEnter && tExit > 0.0f)
        return tEnter;
    else
        return 1e30f; // 충돌 없음
}

__device__ bool traceRay(
    const Ray &r, RayPayload &payload,
    const Triangle *triangles,
    const BVHNode *bvhNodes,    // 추가됨
    const uint32_t *triIndices, // 추가됨
    int numMaterials,           // (사용 안 함, 형식 유지용)
    float tMin, float tMax)
{
    bool anyHit = false;
    float closestSoFar = tMax;
    payload.t = tMax;

    // BVH 순회를 위한 스택 (최대 깊이 64면 충분)
    int stack[64];
    int stackPtr = 0;

    // 루트 노드(0번) 푸시
    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        int nodeIdx = stack[--stackPtr];
        const BVHNode &node = bvhNodes[nodeIdx];

        // 1. 리프 노드인 경우: 포함된 삼각형들과 교차 검사
        if (node.isLeaf())
        {
            for (int i = 0; i < node.primCount; i++)
            {
                // 간접 인덱싱: triIndices를 거쳐 실제 삼각형 접근
                uint32_t triIdx = triIndices[node.firstPrimIdx + i];
                const Triangle &tri = triangles[triIdx];

                RayPayload tmpPayload;
                // intersect 내부에서 t < closestSoFar 검사를 수행하므로 최적화됨
                if (tri.intersect(r, tMin, closestSoFar, tmpPayload))
                {
                    anyHit = true;
                    closestSoFar = tmpPayload.t;
                    payload = tmpPayload;
                }
            }
        }
        // 2. 내부 노드인 경우: 자식 노드 AABB 검사 후 스택에 푸시
        else
        {
            int child1Idx = node.leftChild;
            int child2Idx = node.leftChild + 1; // 인접해 있다고 가정 (구축 로직 확인 필요)

            const BVHNode &child1 = bvhNodes[child1Idx];
            const BVHNode &child2 = bvhNodes[child2Idx];

            float dist1 = intersectAABB(r, child1.aabb);
            float dist2 = intersectAABB(r, child2.aabb);

            // 가까운 자식을 나중에 방문하도록 스택에 '먼 것'부터 넣음
            // (스택은 LIFO이므로 나중에 넣은 게 먼저 나옴)
            if (dist1 > dist2)
            {
                // child2가 더 가까움 -> child1 먼저 푸시, child2 나중에 푸시
                if (dist1 < closestSoFar)
                    stack[stackPtr++] = child1Idx;
                if (dist2 < closestSoFar)
                    stack[stackPtr++] = child2Idx;
            }
            else
            {
                // child1이 더 가까움 -> child2 먼저 푸시, child1 나중에 푸시
                if (dist2 < closestSoFar)
                    stack[stackPtr++] = child2Idx;
                if (dist1 < closestSoFar)
                    stack[stackPtr++] = child1Idx;
            }
        }
    }

    return anyHit;
}

__device__ float3 radiance(
    Ray &r, int maxDepth,
    const Triangle *triangles,
    const BVHNode *bvhNodes,    // 추가
    const uint32_t *triIndices, // 추가
    const Material *materials, int numMaterials,
    curandState *randState)
{
    float3 color = make_float3(0.0f);
    float3 thp = make_float3(1.0f);
    Ray ray = r;

    for (int depth = 0; depth < maxDepth; depth++)
    {
        RayPayload payload;
        if (traceRay(ray, payload, triangles, bvhNodes, triIndices, numMaterials, 0.001f, 1e20f))
        {
            const Material &mat = materials[payload.mIdx];
            color += thp * mat.emission;

            if (mat.type == eDIFFUSE_LIGHT)
                break;

            float3 target = payload.n + randomInUnitSphere(randState);
            if (dot(target, target) < 1e-8f)
                target = payload.n;

            Ray outRay(payload.p, normalize(target));
            thp = thp * mat.albedo;
            ray = outRay;
        }
        else
        {
            float3 unitDir = normalize(ray.dir);
            float t = 0.5f * (unitDir.y + 1.0f);
            float3 skyColor = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
            color += thp * skyColor;
            break;
        }
    }
    return color;
}

__global__ void render(
    float3 *fb,
    int width, int height, int spp,
    Camera cam,
    Triangle *triangles,
    BVHNode *bvhNodes,    // 추가: Device 포인터
    uint32_t *triIndices, // 추가: Device 포인터
    Material *materials, int numMaterials,
    curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= width) || (j >= height))
        return;

    int pIdx = j * width + i;
    curandState localRandState = randState[pIdx];
    float3 color = make_float3(0.0f);

    for (int s = 0; s < spp; s++)
    {
        float u = (float(i) + curand_uniform(&localRandState)) / float(width);
        float v = (float(j) + curand_uniform(&localRandState)) / float(height);

        Ray ray = Ray(cam.origin, normalize(cam.lowerLeftCorner + u * cam.horizontal + v * cam.vertical - cam.origin));
        color += radiance(ray, 5, triangles, bvhNodes, triIndices, materials, numMaterials, &localRandState);
    }

    color /= float(spp);
    fb[pIdx] = color;
    randState[pIdx] = localRandState;
}

int main(int argc, char **argv)
{
    std::cout << "Hello world!" << std::endl;
    std::string inputPath = (argc > 1) ? argv[1] : "./data/CornellBox/CornellBox-Original.obj";
    std::filesystem::path objPath(inputPath);

    if (!std::filesystem::exists(objPath))
    {
        std::cerr << "[Error]: File not found - " << inputPath << std::endl;
        return -1;
    }

    std::string objFp = objPath.string();
    std::string mtlDir = objPath.parent_path().string() + "/";

    std::cout << "[Info] OBJ Path: " << objFp << std::endl;
    std::cout << "[Info] MTL Dir : " << mtlDir << std::endl;

    int width = 1000;
    int height = 1000;
    float aspectRatio = float(width) / float(height);
    Scene scn = loadScene(objFp, mtlDir);

    std::cout << "[Info] Building BVH..." << std::endl;
    BVH hostBVH;
    BuildBVH(scn, hostBVH);

    // Camera cam(
    //     make_float3(0.0f, 1.0f, 4.0f),
    //     make_float3(0.0f, 1.0f, 0.0f),
    //     make_float3(0.0f, 1.0f, 0.0f),
    //     40.0f,
    //     aspectRatio);

    Camera cam(
        make_float3(0.0f, 0.1f, 0.5f), // origin: Y를 0.1로 낮춤 (물체 눈높이)
        make_float3(0.0f, 0.1f, 0.0f), // lookAt: Y를 0.1로 낮춤 (물체 중심 조준)
        make_float3(0.0f, 1.0f, 0.0f), // vup: 그대로 유지 (Y축이 위쪽)
        40.0f,
        aspectRatio);

    int numTriangles = scn.triangles.size();
    int numMaterials = scn.materials.size();
    int numNodes = hostBVH.nodeUsed;

    size_t triangleMemSize = numTriangles * sizeof(Triangle);
    size_t materialMemSize = numMaterials * sizeof(Material);
    size_t nodeMemSize = numNodes * sizeof(BVHNode);
    size_t indexMemSize = numTriangles * sizeof(uint32_t);
    size_t fbMemSize = width * height * sizeof(float3);
    size_t randStateSize = width * height * sizeof(curandState);

    Triangle *dTriangles;
    Material *dMaterials;
    BVHNode *dNodes;
    uint32_t *dIndices;
    float3 *dFrameBuffer;
    curandState *dRandState;

    cudaMalloc((void **)&dTriangles, triangleMemSize);
    cudaMalloc((void **)&dMaterials, materialMemSize);
    cudaMalloc((void **)&dNodes, nodeMemSize);
    cudaMalloc((void **)&dIndices, indexMemSize);
    cudaMalloc((void **)&dFrameBuffer, fbMemSize);
    cudaMalloc((void **)&dRandState, randStateSize);

    cudaMemcpy(dTriangles, scn.triangles.data(), triangleMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMaterials, scn.materials.data(), materialMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dNodes, hostBVH.nodes, nodeMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dIndices, hostBVH.triIndices, indexMemSize, cudaMemcpyHostToDevice);

    int tx = 8, ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    initRandState<<<blocks, threads>>>(width, height, dRandState);
    cudaDeviceSynchronize();

    render<<<blocks, threads>>>(
        dFrameBuffer, width, height, 4096,
        cam,
        dTriangles,
        dNodes,
        dIndices,
        dMaterials, numMaterials,
        dRandState);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    std::vector<float3> fb(width * height);
    cudaMemcpy(fb.data(), dFrameBuffer, fbMemSize, cudaMemcpyDeviceToHost);

    std::cout << "[Info] Processing and saving image..." << std::endl;
    download("output.png", width, height, fb.data());

    hostBVH.free();

    cudaFree(dTriangles);
    cudaFree(dMaterials);
    cudaFree(dNodes);
    cudaFree(dIndices);
    cudaFree(dFrameBuffer);
    cudaFree(dRandState);

    return 0;
}