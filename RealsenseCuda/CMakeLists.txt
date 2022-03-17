# CMakeList.txt: CMake-Projekt für "RealsenseCuda". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.18)
project (RealsenseCuda LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CUDA_ARCHITECTURES 61 75)
# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (RealsenseCuda "RealsenseCuda.cu" "RealsenseCuda.h" "virtual_output.h"  "queue/shared-memory-queue.c" "queue/shared-memory-queue.h" "image_formats.h")
find_package(CUDA REQUIRED 11.2)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(LIBJPEG-TURBO REQUIRED)

#find_package(GLFW3 REQUIRED)
#message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")



target_compile_definitions(RealsenseCuda PUBLIC CUDA=1)

enable_language(CUDA)

# Only compile for newer Pascal cards (to save time)
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")

# Hack to get intellisense working for CUDA includes
if (MSVC)
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);")
endif()

set_target_properties(RealsenseCuda PROPERTIES
                                    CMAKE_CUDA_STANDARD 11
                                    POSITION_INDEPENDENT_CODE ON
                                    CUDA_SEPARABLE_COMPILATION ON
                                    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(RealsenseCuda PUBLIC 
    "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\include"
    "..\\src\\include"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPENGL_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS})
        
target_link_libraries(RealsenseCuda PUBLIC 
    ${CUDA_LIBRARY}
    ${OPENGL_LIBRARY}
    ${OpenCV_LIBS}
    glfw
    GLEW::GLEW
    "C:\\Users\\Joshua\\source\\repos\\RealsenseCuda\\src\\out\\Debug\\yuv.lib"
    "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x64\\realsense2.lib")

#set_target_properties( myapp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "C:\Program Files (x86)\Intel RealSense SDK 2.0\bin\x64" )


