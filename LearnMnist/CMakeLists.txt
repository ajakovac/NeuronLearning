cmake_minimum_required (VERSION 3.0)
project (learn_mnist)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
include_directories("${CMAKE_SOURCE_DIR}/include")

add_executable(learn_mnist learn_mnist.cpp)
add_executable(learn_mnist_CN learn_mnist_CN.cpp)
add_executable(learn_mnist_test learn_mnist_test.cpp)
add_executable(learn_mnist_CN_test learn_mnist_CN_test.cpp)

