cmake_minimum_required(VERSION 3.24)
project(ai_colorbot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── FetchContent dependencies ──────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    SimpleIni
    GIT_REPOSITORY https://github.com/brofield/simpleini.git
    GIT_TAG        v4.22
)
FetchContent_MakeAvailable(SimpleIni)

# ── Find packages ──────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(TensorRT    REQUIRED)
find_package(CUDNN       REQUIRED)

# Serial (wjwwood/serial) must be built separately before configuring this project.
# Clone https://github.com/wjwwood/serial, build it with CMake, then set either:
#   -DSERIAL_ROOT=<install-prefix>   or   SERIAL_ROOT env var
# Alternatively, place the built tree under third_party/serial/ (see FindSerial.cmake).
find_package(Serial      REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS
    core imgproc highgui
    cudaarithm cudawarping cudaimgproc
)

# ── Source files ───────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/capture/capture.cpp
    src/capture/duplicationApiCapture.cpp
    src/config/config.cpp
    src/detection/detector.cpp
    src/detection/postProcess.cpp
    src/inference/tensorrtEngine.cpp
    src/input/target.cpp
    src/input/makcuConnection.cpp
    src/input/mouse.cpp
    src/overlay/overlay.cpp
    src/overlay/visuals.cpp
    src/overlay/drawAim.cpp
    src/overlay/drawCapture.cpp
    src/overlay/drawAi.cpp
    src/overlay/drawColor.cpp
    src/util/otherTools.cpp
    # ImGui (vendored)
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/imgui_impl_win32.cpp
    third_party/imgui/imgui_impl_opengl2.cpp
)

add_executable(ai_colorbot ${SOURCES})

# ── Include directories ────────────────────────────────────────────────────────
target_include_directories(ai_colorbot PRIVATE
    src/
    third_party/
    third_party/imgui/
    ${TensorRT_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Serial_INCLUDE_DIRS}
    ${simpleini_SOURCE_DIR}
)

# ── Link libraries ─────────────────────────────────────────────────────────────
target_link_libraries(ai_colorbot PRIVATE
    ${TensorRT_LIBRARIES}
    ${CUDNN_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    ${OpenCV_LIBS}
    ${Serial_LIBRARIES}
    opengl32
    Ws2_32
    winmm
    dxgi
    d3d11
)

# ── Compile definitions ────────────────────────────────────────────────────────
target_compile_definitions(ai_colorbot PRIVATE
    _WIN32_WINNT=0x0A00  # Windows 10
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

# ── MSVC-specific flags ────────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(ai_colorbot PRIVATE /W3 /MP /utf-8)
    # Suppress C4996 (deprecated POSIX names used by serial library)
    target_compile_definitions(ai_colorbot PRIVATE _CRT_SECURE_NO_WARNINGS)
    # Subsystem: console for debug output
    set_target_properties(ai_colorbot PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:CONSOLE"
    )
endif()

# ── Output directories ─────────────────────────────────────────────────────────
set_target_properties(ai_colorbot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
