# NVCC infers host/device execution spaces for special members defaulted on
# their first declaration. Older Eigen headers annotate them explicitly anyway,
# which causes warning 20012. Patch those declarations in a build-local copy;
# keep both the installed headers and compiler diagnostics unchanged.
function(tbr_fix_eigen_cuda_annotations target)
    if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        return()
    endif()

    get_target_property(eigen_include_dirs Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    foreach(eigen_include_dir IN LISTS eigen_include_dirs)
        set(storage_header "${eigen_include_dir}/Eigen/src/Core/DenseStorage.h")
        if(NOT EXISTS "${storage_header}")
            continue()
        endif()

        file(READ "${storage_header}" storage_source)
        string(REGEX REPLACE
            "EIGEN_DEVICE_FUNC[ \t]+(DenseStorage[^\r\n]*= default;)"
            "\\1" patched_storage_source "${storage_source}")
        if(storage_source STREQUAL patched_storage_source)
            return()
        endif()

        set(patched_include_dir "${CMAKE_CURRENT_BINARY_DIR}/eigen-cuda")
        # Leave the patched header in place so reconfiguring does not touch its
        # timestamp and force all CUDA sources to rebuild.
        file(COPY "${eigen_include_dir}/Eigen" DESTINATION "${patched_include_dir}"
            PATTERN "DenseStorage.h" EXCLUDE)
        if(EXISTS "${eigen_include_dir}/unsupported")
            file(COPY "${eigen_include_dir}/unsupported" DESTINATION "${patched_include_dir}")
        endif()
        file(CONFIGURE
            OUTPUT "${patched_include_dir}/Eigen/src/Core/DenseStorage.h"
            CONTENT "${patched_storage_source}" @ONLY)
        target_include_directories(${target} BEFORE PUBLIC
            "$<BUILD_INTERFACE:${patched_include_dir}>")
        message(STATUS "Fixed redundant Eigen CUDA annotations in ${patched_include_dir}")
        return()
    endforeach()
endfunction()
