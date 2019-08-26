cmake_minimum_required (VERSION 3.0)
project (NeuronLearning)

set(DATASET_PATH "${CMAKE_SOURCE_DIR}/Datasets")
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
include_directories("${CMAKE_SOURCE_DIR}/include")

set(DATASET_DIR "/home/jakovac/Work/AI/Datasets/" CACHE STRING "Dataset path")
configure_file("${CMAKE_SOURCE_DIR}/include/cmake_variables.hpp.in"
               "${CMAKE_SOURCE_DIR}/include/cmake_variables.hpp")


add_subdirectory(Tests)
add_subdirectory(LearnMnist)
add_subdirectory(NL)
add_subdirectory(LearnZero)
add_subdirectory(TrafficSigns)

