# FindCUDNN.cmake
# Locates cuDNN headers and libraries.
#
# Searches (in order):
#   $ENV{CUDNN_PATH}
#   CUDAToolkit include/lib directories
#   Standard NVIDIA CUDA install prefix
#
# Exports:
#   CUDNN_FOUND
#   CUDNN_INCLUDE_DIRS
#   CUDNN_LIBRARIES

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h cudnn_version.h
    HINTS
        "$ENV{CUDNN_PATH}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
)

find_library(CUDNN_LIBRARY
    NAMES cudnn cudnn8 cudnn9
    HINTS
        "$ENV{CUDNN_PATH}/lib"
        "$ENV{CUDNN_PATH}/lib/x64"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64"
    PATH_SUFFIXES lib lib/x64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
    REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
)

if(CUDNN_FOUND)
    set(CUDNN_INCLUDE_DIRS "${CUDNN_INCLUDE_DIR}")
    set(CUDNN_LIBRARIES    "${CUDNN_LIBRARY}")
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
