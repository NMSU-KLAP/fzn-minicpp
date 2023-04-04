cmake_minimum_required(VERSION 3.18)
project(fzn-minicpp-gpu)

# C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CXX)

# CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "60;70;80")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -lineinfo")
# Silent errors
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 177")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 550")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 815")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 830")
string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 997")
enable_language(CUDA)

# Includes
include_directories(.)
include_directories(${CUDAToolkit_INCLUDE_DIRS})


# Sources
set(SRC_MINICPP
	acstr.cpp
	bitset.cpp
	constraint.cpp
	cont.cpp
	context.cpp
	controller.cpp
	domain.cpp
	fail.cpp
	intvar.cpp
	lex.cpp
	mallocWatch.cpp
	matching.cpp
	regular.cpp
	RuntimeMonitor.cpp
	search.cpp
	solver.cpp
	store.cpp
	table.cpp
	trailable.cpp
	trail.cpp)

set(SRC_FZN
	fz_constraints/bool_array.cpp
	fz_constraints/bool_bin.cpp
	fz_constraints/bool_lin.cpp
	fz_constraints/bool_misc.cpp
	fz_constraints/flatzinc.cpp
	fz_constraints/int_array.cpp
	fz_constraints/int_bin.cpp
	fz_constraints/int_lin.cpp
	fz_constraints/int_tern.cpp
	fz_parser/flatzinc.cpp
	fz_parser/lexer.yy.cpp
	fz_parser/parser.tab.cpp
	fz_parser/registry.cpp)

set(SRC_GPU
	gpu_constraints/all_different.cu
	gpu_constraints/circuit.cu
	gpulib/SqrBitMatrix.cu
	gpulib/BitMatrix.cu
	gpulib/Memory.cu)

set(SRC
	fzn-minicpp.cpp
	${SRC_MINICPP}
	${SRC_FZN}
	${SRC_GPU})


# Executable
add_executable(fzn-minicpp fzn-minicpp.cpp ${SRC_MINICPP} ${SRC_FZN} ${SRC_GPU})
