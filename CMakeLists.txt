cmake_minimum_required(VERSION 3.16.3)
project(mesh_smoothing)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# PMP library settings
set(PMP_BUILD_APPS     OFF CACHE BOOL "")
set(PMP_BUILD_EXAMPLES OFF CACHE BOOL "")
set(PMP_BUILD_TESTS    OFF CACHE BOOL "")
set(PMP_BUILD_DOCS     OFF CACHE BOOL "")
add_subdirectory(external/pmp-library)

# add include directories
include_directories(${PROJECT_SOURCE_DIR}/external/pmp-library/src)
include_directories(${PROJECT_SOURCE_DIR}/external/pmp-library/external/glew/include)
include_directories(${PROJECT_SOURCE_DIR}/external/nanoflann)

# set default compiler flags
if((UNIX OR APPLE) AND CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
endif()

if (EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_WEBGL2=1")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
endif()

# Directories to process
add_subdirectory(src/)