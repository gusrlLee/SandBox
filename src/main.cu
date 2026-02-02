#include <iostream>
#include <filesystem>

#include "math_helper.cuh"
#include "structs.cuh"
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

__device__ float3 randomInUnitSphere(curandState *localRandState) {
    float3 p;
    do {
        // -1.0 ~ 1.0 사이의 랜덤 값 추출
        float r1 = curand_uniform(localRandState);
        float r2 = curand_uniform(localRandState);
        float r3 = curand_uniform(localRandState);
        p = 2.0f * make_float3(r1, r2, r3) - make_float3(1.0f, 1.0f, 1.0f);
    } while (dot(p, p) >= 1.0f); 
    return p;
}

__device__ bool traceRay(
    const Ray &r, RayPayload &payload,
    const Triangle *triangles, int numTriangles,
    const Material *materials, int numMaterials,
    float tMin, float tMax)
{
    RayPayload tmpPayload;
    bool anyHit = false;
    float closestSoFar = tMax;

    for (int i = 0; i < numTriangles; i++)
    {
        if (triangles[i].intersect(r, tMin, closestSoFar, tmpPayload)) 
        {
            anyHit = true;
            closestSoFar = tmpPayload.t;
            payload = tmpPayload;
        }
    }

    return anyHit;
}

__device__ float3 radiance
(
    Ray &r, int maxDepth,
    const Triangle *triangles, int numTriangles,
    const Material *materials, int numMaterials,
    curandState *randState
)
{
    float3 color = make_float3(0.0f);
    float3 thp = make_float3(1.0f);
    Ray ray = r;

    for (int depth = 0; depth < maxDepth; depth++)
    {
        RayPayload payload;
        if (traceRay(ray, payload, triangles, numTriangles, materials, numMaterials, 0.00001f, 1e20f))
        {
            const Material &mat = materials[payload.mIdx];

            // L = L + Le
            color += thp * mat.emission;

            if (mat.type == eDIFFUSE_LIGHT)
            {
                break;
            }

            float3 target = payload.n + randomInUnitSphere(randState);
            if (dot(target, target) < 1e-8f) 
            {
                target = payload.n;
            }

            Ray outRay(payload.p, normalize(target));

            float3 attenuation = mat.albedo;
            thp = thp * attenuation;
            ray = outRay;
        }
        else 
        {
            break;
        }
    }
    return color;
}

__global__ void render(
    float3 *fb,
    int width, int height, int spp,
    Camera cam,
    Triangle *triangles, int numTriangles,
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
        color += radiance(ray, 5, triangles, numTriangles, materials, numMaterials, &localRandState);
    }

    color /= float(spp);
    fb[pIdx] = color;
    randState[pIdx] = localRandState;
}

int main(int argc, char** argv)
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

    Camera cam(
        make_float3(0.0f, 1.0f, 4.0f), 
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f), 
        40.0f, 
        aspectRatio
    );

    int numTriangles = scn.triangles.size();
    int numMaterials = scn.materials.size();
    size_t triangleMemSize = numTriangles * sizeof(Triangle);
    size_t materialMemSize = numMaterials * sizeof(Material);
    size_t fbMemSize = width * height * sizeof(float3);
    size_t randStateSize = width * height * sizeof(curandState);

    Triangle *dTriangles;
    Material *dMaterials;
    float3 *dFrameBuffer;
    curandState *dRandState;

    cudaMalloc((void **)&dTriangles, triangleMemSize);
    cudaMalloc((void **)&dMaterials, materialMemSize);
    cudaMalloc((void **)&dFrameBuffer, fbMemSize);
    cudaMalloc((void **)&dRandState, randStateSize);

    cudaMemcpy(dTriangles, scn.triangles.data(), triangleMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dMaterials, scn.materials.data(), materialMemSize, cudaMemcpyHostToDevice);

    int tx = 8, ty = 8;
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    initRandState<<<blocks, threads>>>(width, height, dRandState);
    cudaDeviceSynchronize();

    render<<<blocks, threads>>>(
        dFrameBuffer, width, height, 20000,
        cam, dTriangles, numTriangles,
        dMaterials, numMaterials,
        dRandState);
    cudaDeviceSynchronize();

    std::vector<float3> fb(width * height);
    cudaMemcpy(fb.data(), dFrameBuffer, fbMemSize, cudaMemcpyDeviceToHost);

    std::cout << "[Info] Processing and saving image..." << std::endl;
    download("output.png", width, height, fb.data());

    cudaFree(dTriangles);
    cudaFree(dMaterials);
    cudaFree(dFrameBuffer);
    cudaFree(dRandState);

    return 0;
}