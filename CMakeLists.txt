cmake_minimum_required(VERSION 3.23)
project(MinkowskiEngine LANGUAGES CUDA CXX)

set(TORCH_CUDA_ARCH_LIST "8.6")
set(CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find CUDA
find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED HINTS
             "/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch")

# Collect all source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Create the CUDA library
add_library(
  ${PROJECT_NAME} SHARED
  src/math_functions_cpu.cpp
  src/math_functions_gpu.cu
  src/coordinate_map_manager.cu
  src/coordinate_map_gpu.cu
  src/convolution_kernel.cu
  src/convolution_gpu.cu
  src/depthwise_convolution_kernel.cu
  src/depthwise_convolution_gpu.cu
  src/convolution_transpose_gpu.cu
  src/pooling_avg_kernel.cu
  src/pooling_max_kernel.cu
  src/local_pooling_gpu.cu
  src/local_pooling_transpose_gpu.cu
  src/global_pooling_gpu.cu
  src/broadcast_kernel.cu
  src/broadcast_gpu.cu
  src/pruning_gpu.cu
  src/interpolation_gpu.cu
  src/spmm.cu
  src/gpu.cu
  src/quantization.cpp
  src/direct_max_pool.cpp)

target_compile_options(
  ${PROJECT_NAME}
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>;-fpermissive)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Set CUDA specific properties
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
             POSITION_INDEPENDENT_CODE ON
             CUDA_ARCHITECTURES 86)

# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC src/3rdparty)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES} Python::Python)
