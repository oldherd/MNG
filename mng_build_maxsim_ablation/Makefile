.DEFAULT_GOAL := all
MNG_INDEX_THREADS ?= 1
include ../config.mk

SRC_DIR := src
HEADERS := $(wildcard $(SRC_DIR)/*.hpp $(SRC_DIR)/common/*.hpp)
BASE_FLAGS := $(MNG_FLAGS) -DMNG_FIXED_TARGET_REVERSE_BATCH=1
CONTROL := $(BUILD_DIR)/indexer_mng_centroidmaxsim_build_control_optimized
BRUTE := $(BUILD_DIR)/indexer_mng_brutemaxsim_build_ablation
BINARIES := $(CONTROL) $(BRUTE)

.PHONY: all full clean
all: $(BINARIES)
full: all

$(CONTROL): VARIANT_FLAGS := -DMNG_BUILD_BRUTE_MAXSIM=0
$(BRUTE): VARIANT_FLAGS := -DMNG_BUILD_BRUTE_MAXSIM=1
$(BINARIES): $(SRC_DIR)/mng_build_maxsim_indexer.cpp $(HEADERS) $(MVBUNDLE_OBJ) Makefile ../config.mk | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(FAISS_CPPFLAGS) -I$(SRC_DIR) $(CXXFLAGS) $(OPENMP_FLAGS) $(BACKEND_FLAGS) $(BASE_FLAGS) $(VARIANT_FLAGS) -MMD -MP -MF $@.d -MT $@ $< $(MVBUNDLE_OBJ) $(LDFLAGS) $(FAISS_LDFLAGS) $(MKL_LDFLAGS) $(FAISS_LIBS) $(LDLIBS) -o $@

clean:
	$(RM) $(BINARIES) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
