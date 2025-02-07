# FindTensorRT.cmake
# Locates NVIDIA TensorRT headers and import libraries.
#
# Searches (in order):
#   $ENV{TRT_PATH}         — set this to your TensorRT root
#   $ENV{TENSORRT_ROOT}
#   Standard NVIDIA install prefix
#
# Exports:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES      (nvinfer + nvonnxparser)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        "$ENV{TRT_PATH}/include"
        "$ENV{TENSORRT_ROOT}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/include"
    PATH_SUFFIXES include
)

find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer nvinfer_10
    HINTS
        "$ENV{TRT_PATH}/lib"
        "$ENV{TENSORRT_ROOT}/lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/lib"
    PATH_SUFFIXES lib
)

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser nvonnxparser_10
    HINTS
        "$ENV{TRT_PATH}/lib"
        "$ENV{TENSORRT_ROOT}/lib"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/lib"
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TensorRT_nvinfer_LIBRARY
        TensorRT_nvonnxparser_LIBRARY
        TensorRT_INCLUDE_DIR
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
    set(TensorRT_LIBRARIES
        "${TensorRT_nvinfer_LIBRARY}"
        "${TensorRT_nvonnxparser_LIBRARY}"
    )
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY)
