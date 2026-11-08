#ifndef __SHARED_STORAGE_CUH__
#define __SHARED_STORAGE_CUH__

#include <cuda_runtime.h>
#include <new>
#include <type_traits>

namespace TbrCudaPnP
{
    // Declare this trivial storage in shared memory, then let one thread start
    // the object's lifetime with construct(). Synchronize before other threads
    // access get(). No destructor is needed when the block finishes.
    template <typename T>
    struct alignas(T) SharedStorage
    {
        static_assert(std::is_trivially_destructible<T>::value,
                      "Shared objects must not require destruction");

        unsigned char data[sizeof(T)];

        __device__ T &construct()
        {
            return *::new (static_cast<void *>(data)) T;
        }

        __device__ T &get()
        {
            return *reinterpret_cast<T *>(data);
        }
    };
}

#endif
