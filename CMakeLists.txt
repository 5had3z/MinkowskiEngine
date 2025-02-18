cmake_minimum_required(VERSION 3.23)
project(MinkowskiEngine LANGUAGES CUDA CXX)

# Find CUDA
find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED HINTS "/home/bryce/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch")

# Collect all source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

# Create the CUDA library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_compile_options(${PROJECT_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>;-fpermissive
)

target_compile_features(${PROJECT_NAME}
    PUBLIC
        cxx_std_17
)

# Set CUDA specific properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        src/3rdparty
)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${TORCH_LIBRARIES} Python::Python)
