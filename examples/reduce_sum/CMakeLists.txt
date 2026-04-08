cmake_minimum_required(VERSION 3.14)
project(reduce_sum CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ascendc REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${ascendc_INCLUDE_DIRS}
)

add_executable(reduce_sum reduce_sum.cpp)

target_link_libraries(reduce_sum
    ascendc
)

target_compile_definitions(reduce_sum PRIVATE _ATVOSS_ARCH35_)