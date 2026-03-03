#ifndef SANDBOX_COMMON_CUDA_HEADER
#define SANDBOX_COMMON_CUDA_HEADER

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cmath>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "tinyobjloader/tiny_obj_loader.h"

#include "math.cuh"

struct Camera
{
    float3 origin;
    float3 lowerLeftCorner;
    float3 horizontal;
    float3 vertical;

    __host__ __device__ Camera() {}
    __host__ __device__ Camera(float3 lookFrom, float3 lookAt, float3 vup, float vfov, float aspectRatio)
    {
        float theta = vfov * (PI / 180.0f);
        float h = tan(theta / 2.0f);

        float vHight = 2.0f * h;
        float vWidth = aspectRatio * vHight;

        float3 w = normalize(lookFrom - lookAt);
        float3 u = normalize(cross(vup, w));
        float3 v = cross(w, u);

        origin = lookFrom;
        horizontal = vWidth * u;
        vertical = vHight * v;

        lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - w;
    }
};

struct Ray
{
    float3 orig, dir;
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const float3 &origin, const float3 &direction) : orig(origin), dir(direction) {}
    __host__ __device__ float3 at(float t) const { return orig + t * dir; }
};

struct RayPayload
{
    float t;       // t-value of intersection
    float3 p;      // intersected point
    float3 n;      // normal vector of intersected point
    uint32_t mIdx; // material index
};

struct HitQueue
{
    int triIdx;
    float t;
    float u, v;
};

struct RayWorkItem
{
    int pixelIndex;
    float3 origin;
    float3 direction;
};

struct PixelState
{
    float3 throughput;
    float3 accumulatedColor;
    int depth;
};

struct Triangle
{
    float3 v0, v1, v2, e1, e2, n;
    uint32_t matId;

    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(const float3 &v0, const float3 &v1, const float3 &v2, const uint32_t &materialId) : v0(v0), v1(v1), v2(v2), matId(materialId)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        n = normalize(cross(e1, e2));
    }
    __host__ __device__ float3 normal() const { return n; }
    __host__ __device__ float3 edge1() const { return e1; }
    __host__ __device__ float3 edge2() const { return e2; }
    __host__ __device__ float3 centroid() const { return (v0 + v1 + v2) / 3.0f; }
    __host__ __device__ float area() const { return length(cross(e1, e2)) / 2.0f; }
};

struct AABB
{
    float3 min, max;

    __host__ __device__ AABB()
    {
        min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }

    __host__ __device__ AABB(float3 a, float3 b) : min(a), max(b) {}

    __host__ __device__ void grow(const AABB &other)
    {
        min.x = fminf(min.x, other.min.x);
        min.y = fminf(min.y, other.min.y);
        min.z = fminf(min.z, other.min.z);
        max.x = fmaxf(max.x, other.max.x);
        max.y = fmaxf(max.y, other.max.y);
        max.z = fmaxf(max.z, other.max.z);
    }

    __host__ __device__ void grow(float3 p) 
    {
        min.x = fminf(min.x, p.x);
        min.y = fminf(min.y, p.y);
        min.z = fminf(min.z, p.z);
        max.x = fmaxf(max.x, p.x);
        max.y = fmaxf(max.y, p.y);
        max.z = fmaxf(max.z, p.z);
    }

    __host__ __device__ float area()
    {
		float3 e = max - min; // box extent
		return e.x * e.y + e.y * e.z + e.z * e.x;
	}
};

struct BVHNode 
{
    AABB aabb;
    int leftChild;
    int firstPrimIdx;
    int primCount; // primitive count 

    __host__ __device__ bool isLeaf() const { return primCount > 0; }
};

struct BVH 
{
    BVHNode* nodes;
    uint32_t* triIndices;
    uint32_t nodeUsed = 0;

    void allocate(int numTriangles) 
    {
        nodes = new BVHNode[numTriangles * 2];
        triIndices = new uint32_t[numTriangles];
        nodeUsed = 0;
    }

    void free() 
    {
        if (nodes) delete[] nodes;
        if (triIndices) delete[] triIndices;
        nodes = nullptr;
        triIndices = nullptr;
    }
};

enum MaterialType
{
    eLAMBERTIAN,
    eDIELECTRIC,
    eSPECULAR,
    eDIFFUSE_LIGHT,
};

struct Material
{
    MaterialType type;
    float3 albedo, emission;
    __host__ __device__ Material() : albedo(make_float3(0, 0, 0)), emission(make_float3(0, 0, 0)) {}
};

struct Scene
{
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

__host__ Scene loadScene(const std::string &fp, const std::string &mtlFp)
{
    Scene scn;
    tinyobj::ObjReaderConfig cfg;
    cfg.mtl_search_path = mtlFp;
    cfg.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(fp, cfg))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "TinyObjReader Error: " << reader.Error() << std::endl;
        }
        return scn;
    }

    if (!reader.Warning().empty())
    {
        std::cerr << "TinyObjReader Warning: " << reader.Warning() << std::endl;
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    for (const auto &tMat : materials)
    {
        Material m;
        m.albedo = make_float3(tMat.diffuse[0], tMat.diffuse[1], tMat.diffuse[2]);
        m.emission = make_float3(tMat.emission[0], tMat.emission[1], tMat.emission[2]);

        // for checking light
        float emissionIntensity = m.emission.x + m.emission.y + m.emission.z;

        if (emissionIntensity > 1e-4f)
        {
            m.type = eDIFFUSE_LIGHT;
        }
        else if (tMat.illum == 5)
        {
            m.type = eSPECULAR;
        }
        else if (tMat.illum == 7)
        {
            m.type = eDIELECTRIC;
        }
        else
        {
            m.type = eLAMBERTIAN;
        }

        scn.materials.push_back(m);
    }

    if (scn.materials.empty())
    {
        std::cerr << "Materials of scene is empty." << std::endl;
        Material defaultMat;
        defaultMat.albedo = make_float3(0.7f, 0.7f, 0.7f); // 밝은 회색
        defaultMat.emission = make_float3(0.0f, 0.0f, 0.0f);
        defaultMat.type = eLAMBERTIAN;
        scn.materials.push_back(defaultMat);
    }

    for (const auto &shape : shapes)
    {
        size_t idxOffset = 0;
        for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); face++)
        {
            int fv = shape.mesh.num_face_vertices[face];

            float3 vertices[3];
            for (int i = 0; i < 3; i++)
            {
                tinyobj::index_t idx = shape.mesh.indices[idxOffset + i];

                vertices[i] = make_float3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]);
            }

            int matId = shape.mesh.material_ids[face];
            if (matId < 0 || matId >= scn.materials.size())
            {
                matId = 0;
            }

            scn.triangles.push_back(Triangle(vertices[0], vertices[1], vertices[2], matId));
            idxOffset += fv;
        }
    }
    return scn;
}

void download(std::string filename, int width, int height, int spp, float3 *data)
{
    std::cout << "[Info] Processing and saving image..." << std::endl;
    std::vector<unsigned char> image(width * height * 3);

    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            size_t pixel_index = (height - 1 - j) * width + i;
            float3 pixel = data[pixel_index];

            float scale = 1.0f / float(spp);
            pixel.x *= scale;
            pixel.y *= scale;
            pixel.z *= scale;

            float r = sqrt(pixel.x);
            float g = sqrt(pixel.y);
            float b = sqrt(pixel.z);

            // Gamma Correction
            int ir = int(255.99f * (r > 1.0f ? 1.0f : (r < 0.0f ? 0.0f : r)));
            int ig = int(255.99f * (g > 1.0f ? 1.0f : (g < 0.0f ? 0.0f : g)));
            int ib = int(255.99f * (b > 1.0f ? 1.0f : (b < 0.0f ? 0.0f : b)));

            size_t img_index = (j * width + i) * 3;
            image[img_index + 0] = static_cast<unsigned char>(ir);
            image[img_index + 1] = static_cast<unsigned char>(ig);
            image[img_index + 2] = static_cast<unsigned char>(ib);
        }
    }

    if (stbi_write_png(filename.c_str(), width, height, 3, image.data(), width * 3))
    {
        std::cout << "[Info] Image saved successfully: " << filename << std::endl;
    }
    else
    {
        std::cerr << "[Error] Failed to save image!" << std::endl;
    }
}

#endif