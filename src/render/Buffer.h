#ifndef SANDBOX_BUFFER_HEADER
#define SANDBOX_BUFFER_HEADER

#include <vector>
#include <iostream>

#include <optix.h>
#include <cuda_runtime.h>

class Buffer 
{
public:
	Buffer() : m_DevicePtr(0), m_SizeInBytes(0) {}
    ~Buffer() { free(); }

    void alloc(size_t size)
    {
        if (m_DevicePtr) free();
        cudaMalloc((void**)&m_DevicePtr, size);
        m_SizeInBytes = size;
    }

    void allocAndUpload(const void* data, size_t size)
    {
        alloc(size);
        cudaMemcpy((void*)m_DevicePtr, data, size, cudaMemcpyHostToDevice);
    }

    void upload(const void* data, size_t size)
    {
        if (m_SizeInBytes < size) {
            alloc(size); // 크기가 부족할 때만 새로 할당
        }
        cudaMemcpy((void*)m_DevicePtr, data, size, cudaMemcpyHostToDevice);
    }

    void free()
    {
        if (m_DevicePtr)
        {
            cudaFree((void*)m_DevicePtr);
            m_DevicePtr = 0;
            m_SizeInBytes = 0;
        }
    }

    CUdeviceptr getPtr() const { return (CUdeviceptr)m_DevicePtr; }
    size_t size() const { return m_SizeInBytes; }

private:
	uintptr_t m_DevicePtr;
	size_t m_SizeInBytes;
};


#endif