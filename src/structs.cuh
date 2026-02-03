#ifndef STRUCTS_H
#define STRUCTS_H

#include <iostream>
#include <vector>

#include "math_helper.cuh"
#include "stb/stb_image_write.h"

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

    __device__ bool intersect(const Ray &r, float tMin, float tMax, RayPayload &payload) const
    {
        float3 v0v1 = v1 - v0;
        float3 v0v2 = v2 - v0;
        float3 pvec = cross(r.dir, v0v2);

        float det = dot(v0v1, pvec);

        if (fabs(det) < 1e-8f)
            return false;
        float invDet = 1.0f / det;

        float3 tvec = r.orig - v0;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f)
            return false;

        float3 qvec = cross(tvec, v0v1);
        float v = dot(r.dir, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = dot(v0v2, qvec) * invDet;

        if (t < tMax && t > tMin)
        {
            payload.t = t;
            payload.p = r.at(t);
            payload.n = n; // 미리 계산된 법선 사용 (Flat Shading)
            payload.mIdx = matId;
            return true;
        }

        return false;
    }
    __device__ bool intersectP(const Ray &r, float tMin, float tMax, RayPayload &payload) const
    {
        float3 v0v1 = v1 - v0;
        float3 v0v2 = v2 - v0;
        float3 pvec = cross(r.dir, v0v2);

        float det = dot(v0v1, pvec);

        if (fabs(det) < 1e-8f)
            return false;
        float invDet = 1.0f / det;

        float3 tvec = r.orig - v0;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f)
            return false;

        float3 qvec = cross(tvec, v0v1);
        float v = dot(r.dir, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = dot(v0v2, qvec) * invDet;

        if (t < tMax && t > tMin)
        {
            return true;
        }
        return false;
    }
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

struct BVHNode {
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

#endif