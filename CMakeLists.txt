# 3.8+ for project(LANGUAGES CUDA)
# 3.9+ for OpenMP::OpenMP_CXX
# 3.10+ findopenmp gained support for language-specific components
# 3.11+ for CMake not to add -fopenmp to the nvcc flags
# 3.13+ for target_link_directories

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(azure-pipelines-agent LANGUAGES CUDA VERSION 0.1.0.0)




set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS OFF)





set(CU_SRCS
  src/main.cu
)

add_executable(main ${CU_SRCS})


if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 10)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -code=compute_35,sm_61,sm_70,sm_75")
elseif (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 9)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -code=compute_35,sm_61,sm_70")
elseif (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 8)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -code=compute_35,sm_61")
else()
set(CMAKE_CUDA_FLAGS "-arch=compute_35") # -code implicitly is also compute_35
endif()

