cmake_minimum_required(VERSION 3.20)
project(geochem_sycl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure AdaptiveCpp flags BEFORE find_package
set(ACPP_TARGETS "generic" CACHE STRING "" FORCE)
set(ACPP_EXTRA_ARGS "-O3 --acpp-targets=generic --acpp-export-all" CACHE STRING "" FORCE)

find_package(AdaptiveCpp REQUIRED)

# Get AdaptiveCpp root include directory for JIT headers
get_filename_component(ACPP_ROOT_DIR "${AdaptiveCpp_DIR}/../../.." ABSOLUTE)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ACPP_ROOT_DIR}/include/AdaptiveCpp
)

add_library(sycl_solver_backend SHARED
    src/rates.cpp
    src/jit_compiler.cpp
    src/parser.cpp src/basic_transpiler.cpp
)
add_sycl_to_target(TARGET sycl_solver_backend SOURCES src/rates.cpp src/jit_compiler.cpp src/parser.cpp src/basic_transpiler.cpp)

add_executable(geochem_sycl src/main.cpp)
target_link_libraries(geochem_sycl PRIVATE sycl_solver_backend)
add_sycl_to_target(TARGET geochem_sycl SOURCES src/main.cpp)

add_executable(test_verify tests/test_verify.cpp)
target_link_libraries(test_verify PRIVATE sycl_solver_backend)
add_sycl_to_target(TARGET test_verify SOURCES tests/test_verify.cpp)

