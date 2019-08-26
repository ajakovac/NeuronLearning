cmake_minimum_required (VERSION 3.0)
project (learn_zero)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
set(SFMLModule -lsfml-graphics -lsfml-window -lsfml-system)
include_directories("${CMAKE_SOURCE_DIR}/include")

add_executable(learn_zero learn_zero.cpp)
add_executable(learn_zero_3 learn_zero_3.cpp)
add_executable(random_init random_init.cpp)
add_executable(random_init_pics random_init_pics.cpp)
add_executable(learn_BN learn_BN.cpp)
add_executable(learn_BN_test learn_BN_test.cpp)

target_link_libraries(random_init_pics ${SFMLModule})
