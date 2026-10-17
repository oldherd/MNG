.DEFAULT_GOAL := all
include ../config.mk

SRC_DIR := src
HEADERS := $(wildcard $(SRC_DIR)/*.hpp $(SRC_DIR)/common/*.hpp)
SEARCH_FLAGS := $(MNG_FLAGS) -DMNG_FIXED_TARGET_REVERSE_BATCH=0 $(RERANK_FLAGS)
TARGET := $(BUILD_DIR)/searcher_mng_build_maxsim_r95_quality

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC_DIR)/mng_build_maxsim_r95_searcher.cpp $(SRC_DIR)/mng_build_maxsim_r95_searcher_impl.cpp $(HEADERS) $(MVBUNDLE_OBJ) Makefile ../config.mk | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(FAISS_CPPFLAGS) $(MKL_CPPFLAGS) -I$(SRC_DIR) $(CXXFLAGS) $(OPENMP_FLAGS) $(BACKEND_FLAGS) $(SEARCH_FLAGS) -MMD -MP -MF $@.d -MT $@ $< $(MVBUNDLE_OBJ) $(LDFLAGS) $(FAISS_LDFLAGS) $(MKL_LDFLAGS) $(FAISS_LIBS) $(LDLIBS) -o $@

clean:
	$(RM) $(TARGET) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
