.DEFAULT_GOAL := all
MNG_INDEX_THREADS ?= 1
include ../config.mk

SRC_DIR := src
HEADERS := $(wildcard $(SRC_DIR)/*.hpp $(SRC_DIR)/common/*.hpp)
ABLATION_FLAGS := $(MNG_FLAGS) -DMNG_FIXED_TARGET_REVERSE_BATCH=0
TARGET := $(BUILD_DIR)/searcher_mng_brutemaxsim_ablation

.PHONY: all full clean
all: $(TARGET)
full: all

$(TARGET): $(SRC_DIR)/mng_brutemaxsim_searcher.cpp $(HEADERS) $(MVBUNDLE_OBJ) Makefile ../config.mk | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(FAISS_CPPFLAGS) -I$(SRC_DIR) $(CXXFLAGS) $(OPENMP_FLAGS) $(BACKEND_FLAGS) $(ABLATION_FLAGS) -MMD -MP -MF $@.d -MT $@ $< $(MVBUNDLE_OBJ) $(LDFLAGS) $(FAISS_LDFLAGS) $(MKL_LDFLAGS) $(FAISS_LIBS) $(LDLIBS) -o $@

clean:
	$(RM) $(TARGET) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
