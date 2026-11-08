#ifndef __CUDA_CHECK_H__
#define __CUDA_CHECK_H__

#define CHECK(call)                                                     \
    do                                                                  \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess)                                         \
        {                                                               \
            printf("CUDA error at: '%s:%d' : %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#define ASSERT_WITH_MSG(cond, msg)                          \
    do                                                      \
    {                                                       \
        if (!(cond))                                        \
        {                                                   \
            fprintf(stderr, "Assertion failed: %s\n", msg); \
            assert(cond);                                   \
        }                                                   \
    } while (0)

#endif
