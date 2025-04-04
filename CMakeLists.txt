cmake_minimum_required(VERSION 3.16)

project(particles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Make sure to set -D CMAKE_BUILD_TYPE=Release for max performance!

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(particles particles.cpp)
target_link_libraries(particles PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

option(ENABLE_KOKKOS "Use Kokkos for parallelization" OFF)
if(ENABLE_KOKKOS)
    find_package(Kokkos REQUIRED)
    add_executable(particles_kokkos particles_kokkos.cpp)
    target_link_libraries(particles_kokkos PRIVATE Kokkos::kokkos)
endif()
