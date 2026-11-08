file(GLOB src_inc CONFIGURE_DEPENDS src/*.cu src/*.cpp include/TbrCudaPnP/*.cuh include/TbrCudaPnP/*.h)
add_library(TbrCudaPnP STATIC ${src_inc})
target_include_directories(TbrCudaPnP PUBLIC include)


########################################################################################
################################## Thired-Party-LIB ####################################
########################################################################################


find_package(Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(TbrCudaPnP PUBLIC Eigen3::Eigen)
include(${PROJECT_SOURCE_DIR}/cmake/EigenCudaCompatibility.cmake)
tbr_fix_eigen_cuda_annotations(TbrCudaPnP)



########################################################################################
################################## Thired-Party-LIB ####################################
########################################################################################

target_compile_options(TbrCudaPnP PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(TbrCudaPnP PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(TbrCudaPnP PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)


