cmake_minimum_required(VERSION 3.22)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECHTURES 86;89;120)  # 30xx, 40xx, 50xx series
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_FLAGS_RELEASE "-O2")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart=shared")

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

########################################################################################
########################################################################################
project(Tbr LANGUAGES C CXX CUDA)

# include(MyUsefulFuncs)
add_subdirectory(TbrMain)
add_subdirectory(Module/TbrCudaPnP)
