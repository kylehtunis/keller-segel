cmake_minimum_required(VERSION 3.20)
project(ks_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- pybind11: scikit-build-core injects the cmake dir automatically ---
find_package(pybind11 CONFIG REQUIRED)

# --- Eigen 3.4: header-only, fetched at configure time ---
include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# --- C++ extension module ---
pybind11_add_module(_ks_core
    src/ks_core/ks_solver.cpp
    src/ks_core/bindings.cpp
)

target_include_directories(_ks_core PRIVATE src/ks_core)
target_link_libraries(_ks_core PRIVATE Eigen3::Eigen)

# Aggressive optimisation — safe for a single-machine academic build
target_compile_options(_ks_core PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:
        -O3
        -march=native
        -ffast-math
        -funroll-loops
    >
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)

# Place the .so next to keller_segel.py so `import _ks_core` works without install
set_target_properties(_ks_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
