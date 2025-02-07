# FindSerial.cmake
# Locates wjwwood/serial library (https://github.com/wjwwood/serial).
#
# Searches (in order):
#   $ENV{SERIAL_ROOT}
#   third_party/serial/ (vendored copy)
#
# Exports:
#   Serial_FOUND
#   Serial_INCLUDE_DIRS
#   Serial_LIBRARIES

find_path(Serial_INCLUDE_DIR
    NAMES serial/serial.h
    HINTS
        "$ENV{SERIAL_ROOT}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/serial/include"
    PATH_SUFFIXES include
)

find_library(Serial_LIBRARY
    NAMES serial
    HINTS
        "$ENV{SERIAL_ROOT}/lib"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/serial/build/Release"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/serial/build"
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Serial
    REQUIRED_VARS Serial_LIBRARY Serial_INCLUDE_DIR
)

if(Serial_FOUND)
    set(Serial_INCLUDE_DIRS "${Serial_INCLUDE_DIR}")
    set(Serial_LIBRARIES    "${Serial_LIBRARY}")
endif()

mark_as_advanced(Serial_INCLUDE_DIR Serial_LIBRARY)
