cmake_minimum_required(VERSION 2.8)

project(dlxnet)
set(PROJECT_VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++14")
# Pick the C++ standard to compile with.
# Abseil currently supports C++11, C++14, and C++17.
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
# used for abseil
set(BUILD_SHARED_LIBS TRUE)
# set(GTEST_LINKED_AS_SHARED_LIBRARY TRUE)

add_definitions(-DPLATFORM_POSIX)

option(BUILD_TEST "BUILD TEST" ON)
option(BUILD_DEMO "BUILD DEMO" ON)

find_package(Protobuf REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/dlxnet/core)


# find all files
# *.c
foreach(file_path ${SRC_DIR})
    file(GLOB_RECURSE file_source_c ${file_path}/*.c ${file_path}/*.h)
    set(SRC_FILES ${SRC_FILES} ${file_source_c})
endforeach()
message(STATUS "[*] searching *.c")

# *.cc
foreach(file_path ${SRC_DIR})
    file(GLOB_RECURSE file_source_cc ${file_path}/*.cc ${file_path}/*.hpp)
    set(SRC_FILES ${SRC_FILES} ${file_source_cc})
endforeach()
message(STATUS "[*] searching *.cc")

# *.cpp
foreach(file_path ${SRC_DIR})
    file(GLOB_RECURSE file_source_cpp ${file_path}/*.cpp)
    set(SRC_FILES ${SRC_FILES} ${file_source_cpp})
endforeach()
message(STATUS "[*] searching *.cpp")


# absl
add_subdirectory(./third_party/abseil-cpp)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/abseil-cpp)
set(ABSL_LIBRARIES absl::base absl::synchronization
    absl::strings absl::str_format_internal absl::container
    absl::hash absl::hashtablez_sampler Threads::Threads
    absl::flat_hash_map absl::debugging)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/nsync)
set(NSYNC_LIBRARIES -lnsync -lnsync_cpp)

# in all
# message(${GTEST_LIBRARIES})
set(GTEST_LIBRARIES /usr/lib/libgtest.so)
set(GTEST_MAIN_LIBRARIES /usr/lib/libgtest_main.so)
set(DLXNET_DEPEND -pthread ${Protobuf_LIBRARIES}
    ${ABSL_LIBRARIES} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES}
    -ldl -ldouble-conversion ${NSYNC_LIBRARIES})


# build lib
add_library(dlxnet SHARED ${SRC_FILES})
set_target_properties(dlxnet PROPERTIES VERSION ${PROJECT_VERSION})
target_link_libraries(dlxnet ${DLXNET_DEPEND})


# build demo
if (BUILD_DEMO)
    add_executable(main ${CMAKE_CURRENT_LIST_DIR}/dlxnet/example/demo_main.cc)
    target_link_libraries(main dlxnet)
endif()

# build test
if (BUILD_TEST)
    add_executable(test_main ${CMAKE_CURRENT_LIST_DIR}/dlxnet/example/test_main.cc)
    target_link_libraries(test_main dlxnet)
endif()
