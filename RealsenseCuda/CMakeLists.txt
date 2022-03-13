# CMakeList.txt: CMake-Projekt für "RealsenseCuda". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.8)
project (RealsenseCuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (RealsenseCuda "RealsenseCuda.cu" "RealsenseCuda.h")
find_package(CUDA REQUIRED 11.2)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED PATHS "C:\\opencv\\build\\x64\\vc14\\lib")

#find_package(GLFW3 REQUIRED)
#message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

set(CMAKE_CUDA_ARCHITECTURES 61 75)

  target_compile_definitions(RealsenseCuda PUBLIC CUDA=1)

  enable_language(CUDA)

  # Only compile for newer Pascal cards (to save time)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")

  # Hack to get intellisense working for CUDA includes
  if (MSVC)
    set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath);")
  endif()

  set_target_properties(RealsenseCuda PROPERTIES
                                     CMAKE_CUDA_STANDARD 17
                                     POSITION_INDEPENDENT_CODE ON
                                     CUDA_SEPARABLE_COMPILATION ON
                                     CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  target_include_directories(RealsenseCuda PUBLIC 
        "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\include"
        "C:\\glew\\include"
        "C:\\glfw\\include"
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${OPENGL_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS})
        
  target_link_libraries(RealsenseCuda PUBLIC 
        ${CUDA_LIBRARY}
        ${OPENGL_LIBRARY}
        ${OpenCV_LIBS}
        "C:\\glew\\lib\\Release\\x64\\glew32.lib"
        "C:\\glfw\\lib-vc2022\\glfw3.lib"
        "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x64\\realsense2.lib")

  #set_target_properties( myapp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "C:\Program Files (x86)\Intel RealSense SDK 2.0\bin\x64" )


