cmake_minimum_required(VERSION 3.0)
project(NVIDIA_SGEMM_PRACTICE)

# gcc/g++ compilation parameter description:
# -O1~3 4 levels of compiler optimization options, -O1 defaults, the larger the level, the better the optimization effect, but the longer the compilation time;
# -std=c++11, compiled using C++11 standard
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

# nvcc compilation parameter description:
# -g: Host code adds debugging information;
# -G: The device code generates debugging information, which will disable most compiler optimizations, causing the device code to run slowly;
# -Xptxas -dlcm=ca enables L1 cache, -Xptxas -dlcm=cg turns off L1 cache

# set(CUDA_NVCC_FLAGS -g;-G;-Xptxas;-dlcm=ca)
# set(CUDA_NVCC_FLAGS -Xptxas;-dlcm=cg)
set(CUDA_NVCC_FLAGS -arch=compute_70;-code=compute_70)

# If FIND CUDA ERROR, add configuration environment variables and dynamic library paths in ~/.bashrc
#CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
find_package(CUDA REQUIRED)

#Configure header file search path
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
#Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
# Executable file output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# Generate executable file
CUDA_ADD_EXECUTABLE(sgemm sgemm.cu ${SRC})

# link cudart cublas
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
