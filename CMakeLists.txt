cmake_minimum_required(VERSION 3.20)
project(poker_assistant_native LANGUAGES CXX)

# Use C++17 to match the source: arrays, structured bindings, std::array
# initialisers. The .cpp uses no features beyond C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release-mode optimisation by default. CMake leaves CMAKE_BUILD_TYPE
# empty on single-config generators (Make, Ninja); without this users
# would get an unoptimised debug build, which defeats the point of a
# native backend.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# scikit-build-core makes Python and pybind11 available via find_package;
# this is the recommended path documented in the scikit-build-core README.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# The extension target. The name must match the PYBIND11_MODULE() name
# in the .cpp source: import poker_assistant.compute._native_equity will
# resolve to this compiled artefact.
pybind11_add_module(
    _native_equity
    MODULE
    src/poker_assistant/compute/_native_equity.cpp
)

# Per-target tuning:
#   * -O3 and -fvisibility=hidden are pybind11 best practice; the latter
#     keeps the shared object's exported symbol set minimal.
#   * Warnings are surfaced (-Wall -Wextra) but not fatal: the source is
#     external and we do not gate builds on its lint cleanliness.
target_compile_options(
    _native_equity
    PRIVATE
        -O3
        -Wall
        -Wextra
        -fvisibility=hidden
)

# scikit-build-core picks up the install destination from the
# [tool.scikit-build] section of pyproject.toml; here we only declare
# what to install and where, relative to the wheel root.
install(
    TARGETS _native_equity
    LIBRARY DESTINATION poker_assistant/compute
)
