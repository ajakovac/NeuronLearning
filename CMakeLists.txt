cmake_minimum_required (VERSION 3.0)
project (NeuronLearning)

set(DATASET_PATH "${CMAKE_SOURCE_DIR}/Datasets")
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
include_directories("${CMAKE_SOURCE_DIR}/include")

add_subdirectory(Tests)
add_subdirectory(LearnMnist)

