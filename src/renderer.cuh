#ifndef SANDBOX_RENDERER_CUDA_HEADER
#define SANDBOX_RENDERER_CUDA_HEADER

#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>


#define OPTIX_CHECK(call)                                                                             \
    {                                                                                                 \
        OptixResult res = call;                                                                       \
        if (res != OPTIX_SUCCESS)                                                                     \
        {                                                                                             \
            fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
            exit(2);                                                                                  \
        }                                                                                             \
    }


class Renderer
{
public:
    Renderer();
    ~Renderer();
};

#endif