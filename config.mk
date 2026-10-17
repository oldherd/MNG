# Shared build configuration. Override these variables on the make command line.
ifeq ($(origin CXX),default)
CXX := g++
endif
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=
OPENMP_FLAGS ?= -fopenmp
BUILD_DIR ?= build
MVBUNDLE_DIR := ../encoders-v2
MVBUNDLE_OBJ := $(BUILD_DIR)/mvbundle.o

FAISS_ROOT ?=
MKL_ROOT ?=
OPENBLAS_ROOT ?=
EIGEN_INCLUDE_DIR ?= /usr/include/eigen3
FAISS_LIB_DIR ?= $(FAISS_ROOT)/lib
MKL_LIB_DIR ?= $(if $(wildcard $(MKL_ROOT)/lib/intel64/libmkl_core.*),$(MKL_ROOT)/lib/intel64,$(MKL_ROOT)/lib)
OPENBLAS_LIB_DIR ?= $(OPENBLAS_ROOT)/lib

FAISS_CPPFLAGS = $(if $(strip $(FAISS_ROOT)),-I$(FAISS_ROOT)/include)
MKL_CPPFLAGS = $(if $(strip $(MKL_ROOT)),-I$(MKL_ROOT)/include)
FAISS_LDFLAGS = $(if $(strip $(FAISS_ROOT)),-L$(FAISS_LIB_DIR) -Wl$(comma)-rpath$(comma)$(FAISS_LIB_DIR))
MKL_LDFLAGS = $(if $(strip $(MKL_ROOT)),-L$(MKL_LIB_DIR) -Wl$(comma)-rpath$(comma)$(MKL_LIB_DIR))
OPENBLAS_LDFLAGS = $(if $(strip $(OPENBLAS_ROOT)),-L$(OPENBLAS_LIB_DIR) -Wl$(comma)-rpath$(comma)$(OPENBLAS_LIB_DIR))
comma := ,
# Match the original MKL-backed FAISS build; override FAISS_LIBS for other builds.
MKL_LIBS ?= -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl
FAISS_LIBS ?= -lfaiss $(MKL_LIBS)
OPENBLAS_LIBS ?= -lopenblas -lgfortran

MNG_BACKEND ?= auto
MNG_INDEX_THREADS ?= 96
MNG_QUERY_THREADS ?= 1
ifeq ($(MNG_BACKEND),auto)
MNG_BACKEND_DETECTED := $(shell if [ -r /proc/cpuinfo ] && grep -qw avx512f /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then echo avx512; elif [ -r /proc/cpuinfo ] && grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo; then echo avx2; else echo scalar; fi)
else
MNG_BACKEND_DETECTED := $(MNG_BACKEND)
endif
BACKEND_FLAGS :=
ifeq ($(MNG_BACKEND_DETECTED),avx512)
BACKEND_FLAGS += -DMNG_BACKEND_AVX512 -mavx512f -mavx512dq -mavx512vl -mfma
else ifeq ($(MNG_BACKEND_DETECTED),avx2)
BACKEND_FLAGS += -DMNG_BACKEND_AVX2 -mavx2 -mfma
else ifneq ($(MNG_BACKEND_DETECTED),scalar)
$(error MNG_BACKEND must be auto, scalar, avx2, or avx512)
endif

MNG_FLAGS = -DMNG_MODE_EXP \
	-DMNG_INDEX_THREADS=$(MNG_INDEX_THREADS) \
	-DMNG_QUERY_THREADS=$(MNG_QUERY_THREADS) \
	-DMNG_FIXED_SOURCE_FORWARD_BATCH=0 -DMNG_SCORER_PROFILING=0
RERANK_FLAGS = -DMNG_EXACT_RERANK_USE_MKL \
	-DMNG_EXACT_RERANK_VALIDATE=0 -DMNG_EXACT_RERANK_DIAGNOSTICS=0 \
	-DMNG_EXACT_RERANK_TRUST_INTERNAL_CANDIDATES=1 \
	-DMNG_EXACT_RERANK_PRETRANSPOSE_Q32=1 \
	-DMNG_EXACT_RERANK_STAGE_TIMERS=0 -DMNG_EXACT_RERANK_PROGRESS=0

$(BUILD_DIR):
	mkdir -p $@

# Each component owns its object to allow independent and parallel builds.
$(MVBUNDLE_OBJ): $(MVBUNDLE_DIR)/mvbundle.cpp $(MVBUNDLE_DIR)/mvbundle.hpp Makefile ../config.mk | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF $@.d -MT $@ -c $< -o $@

-include $(wildcard $(BUILD_DIR)/*.d)
