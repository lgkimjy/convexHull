# Minimum required CMake version
cmake_minimum_required(VERSION 3.16)

# Project name
project(
    MyProject
    VERSION 0.0.0
    DESCRIPTION "CMake Project Template"
    LANGUAGES CXX
)

find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
  message(STATUS "[Eigen3] Found Eigen3 library.")
else()
  message(STATUS "[Eigen3] Cannot find a package.")
endif()


# Set Compiler flags & options
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)        # LTO option
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")           # -Wall -Wextra -Wpedantic -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -march=native")   # -Ofast, -O3, -O2, -O1, -Os

# Add include directories
include_directories(
    include
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    ${EIGEN3_INCLUDE_DIR}
    DEPENDS system_lib
)

# Add source files
set(ALGORITHMS_SRC
    ${CMAKE_SOURCE_DIR}/include/Foo/foo.cpp
)
# Add executable target
add_executable(project
    src/main.cpp
    ${ALGORITHMS_SRC}
)
# Set include directories
target_include_directories(project PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
# Link Eigen3 library
target_link_libraries(project PRIVATE 
    Eigen3::Eigen
)

############################
######## Unit tests ########
############################
# cmake -DBUILD_TESTING=ON ..
if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED) # brew install googletest (macOS)
    include(GoogleTest)
    add_executable(tests 
        tests/test_main.cpp
        tests/test_foo.cpp
        ${ALGORITHMS_SRC}
    )
    target_link_libraries(tests 
        GTest::gtest
    )
endif()