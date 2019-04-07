cmake_minimum_required (VERSION 3.0)
project (test_all)


set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
set(SFMLModule -lsfml-graphics -lsfml-window -lsfml-system)
include_directories("${CMAKE_SOURCE_DIR}/include")

file(GLOB TEST_SOURCES *.cpp)
foreach( testfile ${TEST_SOURCES})
  get_filename_component( fn ${testfile} NAME_WE)
  add_executable( ${fn} ${testfile})
endforeach( testfile ${TEST_SOURCES}) 

target_link_libraries(test_imagelayer ${SFMLModule})
target_link_libraries(dataset_test ${SFMLModule})

