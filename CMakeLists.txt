cmake_minimum_required(VERSION 3.28)
project(bandit-im CXX)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
)

# Set debug/release flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# Static linking for GCC/Clang runtime
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")

# Include directories
include_directories(include)

# Main program
file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")

add_library(program_lib OBJECT ${SOURCES})

# Main executable
add_executable(bandit-im src/main.cpp)
target_link_libraries(bandit-im PRIVATE program_lib argparse)

# Tests
enable_testing()

add_subdirectory(3rd-party/Catch2)

file(GLOB TEST_SOURCES "tests/test_*.cpp")
add_executable(bandit-im-unit-tests ${TEST_SOURCES})
target_link_libraries(bandit-im-unit-tests PRIVATE Catch2::Catch2WithMain program_lib)

include(CTest)
include(Catch)
catch_discover_tests(bandit-im-unit-tests)
