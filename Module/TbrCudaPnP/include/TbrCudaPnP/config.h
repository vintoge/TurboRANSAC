#ifndef __CONFIG_H__
#define __CONFIG_H__

// Can easily switch precision in CMakeLists.txt
#define USE_FLOAT
// #define USE_DOUBLE

#ifdef USE_FLOAT
typedef float real_t_base;
#elif defined(USE_DOUBLE)
typedef double real_t_base;
#else
#error "Please define USE_FLOAT or USE_DOUBLE"
#endif

namespace TbrCudaPnP
{
    using real_t = real_t_base;

    #ifdef USE_FLOAT
        constexpr real_t eps_refine_lambda = 1e-6f;
        constexpr real_t eps_root2real = 1e-6f;
    #elif defined(USE_DOUBLE)
        constexpr real_t eps_refine_lambda = 1e-10;
        constexpr real_t eps_root2real = 1e-12;
    #endif

}

#endif
