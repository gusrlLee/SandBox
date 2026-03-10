#ifndef SANDBOX_SHADER_COMMON_HEADER
#define SANDBOX_SHADER_COMMON_HEADER

#include <optix.h>
#include <vector_types.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ALIGN(x) __align__(x)
#else
#if defined(_MSC_VER)
#define ALIGN(x) __declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif
#endif

struct LaunchParams 
{
    float4* pbo;
    uint32_t width;
    uint32_t height;
    uchar4* colorBuffer;
};

template <typename T>
struct ALIGN(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};
typedef SbtRecord<EmptyData> RaygenRecord;
typedef SbtRecord<EmptyData> MissRecord;
typedef SbtRecord<EmptyData> HitgroupRecord;

#endif // SANDBOX_SHADER_COMMON_HEADER