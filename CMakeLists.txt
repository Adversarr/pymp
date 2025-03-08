cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Install mathprim here.
list(APPEND CMAKE_PREFIX_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/install/")

project(pymp)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(mathprim REQUIRED)

file(GLOB_RECURSE PYMP_SRC "src/*.cpp")

if(MATHPRIM_ENABLE_CUDA)
    file(GLOB_RECURSE PYMP_CU "src/*.cu")
    list(APPEND PYMP_SRC ${PYMP_CU})

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
  
    enable_language(CUDA)
endif()

nanobind_add_module(libpymp ${PYMP_SRC})

target_link_libraries(libpymp PRIVATE mathprim::mathprim)

# Install the module
install(TARGETS libpymp LIBRARY DESTINATION .)