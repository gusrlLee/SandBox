#ifndef SANDBOX_RTCORE_CUDA_HEADER
#define SANDBOX_RTCORE_CUDA_HEADER

#include "common.cuh"

__device__ bool intersectTriangle(const Triangle &tri, const Ray &r, float tMin, float tMax, RayPayload &payload)
{
    float3 v0v1 = tri.v1 - tri.v0;
    float3 v0v2 = tri.v2 - tri.v0;
    float3 pvec = cross(r.dir, v0v2);

    float det = dot(v0v1, pvec);

    if (fabs(det) < 1e-8f)
        return false;
    float invDet = 1.0f / det;

    float3 tvec = r.orig - tri.v0;
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
        payload.n = tri.n; // 미리 계산된 법선 사용 (Flat Shading)
        payload.mIdx = tri.matId;
        return true;
    }

    return false;
}

#endif