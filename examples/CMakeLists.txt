# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(ATVOSS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(ATVOSS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_custom_target(atvoss_examples)

# arch specific compile options
if(SOC STREQUAL "a2" OR SOC STREQUAL "a3")
    set(NPU_ARCH dav-2201)
elseif(SOC STREQUAL "ascend950")
    set(NPU_ARCH dav-3510)
else()
    message(FATAL_ERROR
        "SOC only supports a2/a3/ascend950, but get ${SOC}"
    )
    set(NPU_ARCH)
endif()

# include directories
set(INCLUDE_DIRECTORIES
    ${ASCEND_INCLUDE_DIRS}
    ${ATVOSS_INCLUDE_DIRS}
    ${ATVOSS_EXAMPLES_COMMON_SOURCE_DIR}
)

# link directories
set(LINK_DIRECTORIES
    ${ASCEND_DIR}/lib64
)

# link libraries
set(LINK_LIBRARIES
    ascendcl
    platform
    register
    tiling_api
    runtime
)

# compile options
set(COMPILE_OPTIONS
    -O3
    -fdiagnostics-color=always
    -w
)

macro(atvoss_example_add_executable NAME)
    # clear CMAKE_CXX_FLAGS to avoid affecting bisheng compile
    unset(CMAKE_CXX_FLAGS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_C_COMPILER ${BISHENG})
    set(CMAKE_CXX_COMPILER ${BISHENG})
    set(CMAKE_LINKER ${BISHENG})

    message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
    set(EXAMPLE_DESTINATION bin)

    # get compile flags for current op
    set(COMPILE_FLAGS "--npu-arch=${NPU_ARCH} -xasc ")
    message(STATUS "COMPILE FLAGS: ${COMPILE_FLAGS}")

    # set_source_files_properties
    set(SOURCE_FILES ${ARGN})
    set_source_files_properties(
    ${SOURCE_FILES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${COMPILE_FLAGS}"
    )

    add_executable(${NAME} ${SOURCE_FILES})

    set_target_properties(${NAME} PROPERTIES
        LINK_DEPENDS_NO_SHARED ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_compile_options(${NAME} PRIVATE ${COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE ${INCLUDE_DIRECTORIES})
    target_link_directories(${NAME} PRIVATE ${LINK_DIRECTORIES})
    target_link_libraries(${NAME} PRIVATE ${LINK_LIBRARIES})

    add_dependencies(atvoss_examples ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT atvoss_examples)
endmacro()

# regular examples
foreach(EXAMPLE
    abs
    rms_norm
    muls
)
    add_subdirectory(${EXAMPLE})
endforeach()
