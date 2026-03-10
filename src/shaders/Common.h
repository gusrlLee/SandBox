#ifndef SANDBOX_SHADER_COMMON_HEADER
#define SANDBOX_SHADER_COMMON_HEADER

#include <vector_types.h>

struct LaunchParams 
{
    float4* pbo;
    int width;
    int height;
};

#endif // SANDBOX_SHADER_COMMON_HEADER