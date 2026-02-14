#include <iostream>
#include <filesystem>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "common.cuh"
#include "renderer.cuh"

int main(int argc, char** argv)
{
    std::cout << "[1/N] Loading Scene... \t";
    std::string inputPath = (argc > 1) ? argv[1] : "./assets/CornellBox/CornellBox-Original.obj";
    std::filesystem::path objPath(inputPath);
    
    if (!std::filesystem::exists(objPath))
    {
        std::cerr << "[Error]: File not found - " << inputPath << std::endl;
        return -1;
    }
    
    std::string objFp = objPath.string();
    std::string mtlDir = objPath.parent_path().string() + "/";

    Scene scene = loadScene(objFp, mtlDir);
    std::cout << "Success!" << std::endl;
    std::cout << "[Info] OBJ Path: " << objFp << std::endl;
    std::cout << "[Info] MTL Dir : " << mtlDir << std::endl;
    std::cout << "Numer of triangles: " << scene.triangles.size() << std::endl;
    std::cout << "Numer of materials: " << scene.materials.size() << std::endl;
    
    std::cout << "[2/N] Setting Scene and Camera... \t";
    int width = 1000;
    int height = 1000;
    float aspectRatio = float(width) / float(height);
    Camera cam(
        make_float3(0.0f, 1.0f, 4.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        40.0f,
        aspectRatio);

    // Camera cam(
    //     make_float3(0.0f, 0.1f, 0.5f), // origin: Y를 0.1로 낮춤 (물체 눈높이)
    //     make_float3(0.0f, 0.1f, 0.0f), // lookAt: Y를 0.1로 낮춤 (물체 중심 조준)
    //     make_float3(0.0f, 1.0f, 0.0f), // vup: 그대로 유지 (Y축이 위쪽)
    //     40.0f,
    //     aspectRatio);

    std::cout << "Success!" << std::endl;
    
    std::cout << "[3/N] Setting RendererState ... \t";
    int numPixels = width * height;
    int numTriangles = scene.triangles.size();
    int numMaterials = scene.materials.size();
    
    Triangle *dTriangles;
    Material *dMaterials;
    float3 *dFrameBuffer;
    curandState *dRandState;
    
    cudaMalloc((void **)&dTriangles, numTriangles * sizeof(Triangle));
    cudaMalloc((void **)&dMaterials, numMaterials * sizeof(Material));
    cudaMalloc((void **)&dFrameBuffer, width * height * sizeof(float3));
    cudaMalloc((void **)&dRandState, width * height * sizeof(curandState));

    cudaMemcpy(dTriangles, scene.triangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(dMaterials, scene.materials.data(), numMaterials * sizeof(Material), cudaMemcpyHostToDevice);
    cudaMemset(dFrameBuffer, 0, width * height * sizeof(float3)); 
    std::cout << "Success!" << std::endl;
    
    std::cout << "[4/N] Setting Wavefront Buffer ... \t";
    RayWorkItem *dQueueCurrent, *dQueueNext; // Ping-Pong Queues
    HitQueue *dHitQueue;
    PixelState *dPixelStates;
    int *dNumRays;      // 현재 큐에 담긴 레이 개수 (Device 변수)
    int *dNextNumRays;  // 다음 큐에 담길 레이 개수 (Device 변수 - Atomic용)

    cudaMalloc((void **)&dQueueCurrent, numPixels * sizeof(RayWorkItem));
    cudaMalloc((void **)&dQueueNext,    numPixels * sizeof(RayWorkItem));
    cudaMalloc((void **)&dHitQueue,     numPixels * sizeof(HitQueue));
    cudaMalloc((void **)&dPixelStates,  numPixels * sizeof(PixelState));
    cudaMalloc((void **)&dNumRays,      sizeof(int));
    cudaMalloc((void **)&dNextNumRays,  sizeof(int));

    cudaMemset(dNumRays, 0, sizeof(int));
    cudaMemset(dNextNumRays, 0, sizeof(int));
    std::cout << "Success!" << std::endl;
    
    std::cout << "[5/N] Rendering ... \t";
    int spp = 4096;
    int maxDepth = 5;

    render(
        dFrameBuffer, width, height, spp, maxDepth, // framebuffer, camera, width, height, spp, maxDepth
        cam, // camera
        dTriangles, numTriangles, // triangles
        dMaterials, numMaterials, // materials
        dQueueCurrent, // current queue
        dQueueNext, // next queue
        dHitQueue, // hit queue
        dPixelStates, // pixel states
        dNumRays, // num rays
        dNextNumRays, // next num rays
        dRandState // rand state
    );
    std::cout << "Success!" << std::endl;
    
    std::cout << "[5/N] Downloading ... \t";
    std::vector<float3> fb(width * height);
    cudaMemcpy(fb.data(), dFrameBuffer, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    download("output.png", width, height, spp, fb.data());
    std::cout << "Success!" << std::endl;

    // Wavefront Buffers Free
    cudaFree(dQueueCurrent);
    cudaFree(dQueueNext);
    cudaFree(dHitQueue);
    cudaFree(dPixelStates);
    cudaFree(dNumRays);
    cudaFree(dNextNumRays);

    cudaFree(dTriangles);
    cudaFree(dMaterials);
    cudaFree(dFrameBuffer);
    cudaFree(dRandState);
    return 0;
}