cmake_minimum_required(VERSION 3.10)

project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA package
#find_package(CUDA 12.1 REQUIRED)
find_package(CUDA REQUIRED)


# Add -O3 optimization flag for g++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add -O3 optimization flag for nvcc
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Find MPI package
set(MPI_ROOT $ENV{MPI_ROOT})
find_package(MPI REQUIRED)


# Find OpenMP package
find_package(OpenMP REQUIRED)

# Include MPI headers
include_directories(${MPI_CXX_INCLUDE_PATH} src)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70)

# Set source files
set(SRC_FILES
    src/main.cu
    src/util.cpp
    src/lsh.cu
    src/param.cu
)

# Add executable
add_executable(main ${SRC_FILES})

# Link MPI and OpenMP
target_link_libraries(main ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
