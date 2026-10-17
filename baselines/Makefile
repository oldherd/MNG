.DEFAULT_GOAL := all
include ../config.mk

INCLUDES := -I. -I.. -I$(MVBUNDLE_DIR) -Ihnswlib
COMMON_HEADERS := $(wildcard ../common/*.hpp hnswlib/*.h)
PARLAYANN_ROOT := third_party/ParlayANN
PARLAYANN_CPPFLAGS := -I$(PARLAYANN_ROOT)/algorithms -I$(PARLAYANN_ROOT)/parlaylib/include
PARLAYANN_LIBS ?= -pthread -ldl -ljemalloc
GEM_ROOT := third_party/gem
GEM_CPPFLAGS := -I$(GEM_ROOT) -I$(GEM_ROOT)/otlib -I$(EIGEN_INCLUDE_DIR) -DGEM_USE_EXTERNAL_CBLAS
GEM_ARCH_FLAGS ?= -mavx -mfma
GEM_OBJ := $(BUILD_DIR)/gem_emd.o

BIN_NAMES := indexer_colbert searcher_colbert indexer_plaid searcher_plaid \
	indexer_dessert searcher_dessert indexer_warp searcher_warp \
	indexer_igp searcher_igp indexer_muvera_diskann searcher_muvera_diskann \
	indexer_gem_official searcher_gem_official \
	searcher_brute
BINARIES := $(addprefix $(BUILD_DIR)/,$(BIN_NAMES))
FAISS_BINS := $(addprefix $(BUILD_DIR)/,indexer_colbert searcher_colbert \
	indexer_plaid indexer_warp indexer_igp indexer_gem_official)
MUVERA_BINS := $(addprefix $(BUILD_DIR)/,indexer_muvera_diskann searcher_muvera_diskann)
GEM_BINS := $(addprefix $(BUILD_DIR)/,indexer_gem_official searcher_gem_official)

.PHONY: all clean official_gem muvera gt $(BIN_NAMES)
all: $(addprefix $(BUILD_DIR)/,$(BIN_NAMES))
official_gem: $(GEM_BINS)
muvera: $(MUVERA_BINS)
gt: $(BUILD_DIR)/searcher_brute

$(FAISS_BINS): EXTRA_CPPFLAGS += $(FAISS_CPPFLAGS)
$(FAISS_BINS): EXTRA_LDFLAGS += $(FAISS_LDFLAGS) $(MKL_LDFLAGS)
$(FAISS_BINS): EXTRA_LIBS += $(FAISS_LIBS)
$(MUVERA_BINS): EXTRA_CPPFLAGS += $(PARLAYANN_CPPFLAGS)
$(MUVERA_BINS): EXTRA_LIBS += $(PARLAYANN_LIBS)
$(GEM_BINS): EXTRA_CPPFLAGS += $(GEM_CPPFLAGS)
$(GEM_BINS): EXTRA_CXXFLAGS += $(GEM_ARCH_FLAGS)
$(GEM_BINS): EXTRA_LDFLAGS += $(OPENBLAS_LDFLAGS)
$(GEM_BINS): EXTRA_LIBS += $(OPENBLAS_LIBS)
$(GEM_BINS): EXTRA_OBJECTS := $(GEM_OBJ)

# Put source prerequisites first: $< must remain the entrypoint source.
define binary
$(BUILD_DIR)/$(1): $(2) $(MVBUNDLE_OBJ) $(wildcard $(dir $(2))*.hpp) $(COMMON_HEADERS) Makefile ../config.mk | $(BUILD_DIR)
	$$(CXX) $$(CPPFLAGS) $$(EXTRA_CPPFLAGS) $$(INCLUDES) $$(CXXFLAGS) $$(OPENMP_FLAGS) $$(EXTRA_CXXFLAGS) -MMD -MP -MF $$@.d -MT $$@ $$< $$(MVBUNDLE_OBJ) $$(EXTRA_OBJECTS) $$(LDFLAGS) $$(EXTRA_LDFLAGS) $$(EXTRA_LIBS) $$(LDLIBS) -o $$@
$(1): $(BUILD_DIR)/$(1)
endef
$(eval $(call binary,indexer_colbert,colbert/colbert_indexer.cpp))
$(eval $(call binary,searcher_colbert,colbert/colbert_searcher.cpp))
$(eval $(call binary,indexer_plaid,plaid/plaid_indexer.cpp))
$(eval $(call binary,searcher_plaid,plaid/plaid_searcher.cpp))
$(eval $(call binary,indexer_dessert,dessert/dessert_indexer.cpp))
$(eval $(call binary,searcher_dessert,dessert/dessert_searcher.cpp))
$(eval $(call binary,indexer_warp,warp/warp_indexer.cpp))
$(eval $(call binary,searcher_warp,warp/warp_searcher.cpp))
$(eval $(call binary,indexer_igp,igp/igp_indexer.cpp))
$(eval $(call binary,searcher_igp,igp/igp_searcher.cpp))
$(eval $(call binary,indexer_muvera_diskann,muvera_diskann/muvera_diskann_indexer.cpp))
$(eval $(call binary,searcher_muvera_diskann,muvera_diskann/muvera_diskann_searcher.cpp))
$(eval $(call binary,indexer_gem_official,official_gem/gem_official_indexer.cpp))
$(eval $(call binary,searcher_gem_official,official_gem/gem_official_searcher.cpp))
$(eval $(call binary,searcher_brute,brute/brute_searcher.cpp))

$(MUVERA_BINS): $(wildcard $(PARLAYANN_ROOT)/algorithms/utils/*.h) $(PARLAYANN_ROOT)/algorithms/vamana/index.h
$(GEM_BINS): $(GEM_OBJ) $(wildcard $(GEM_ROOT)/hnswlib/*.h) $(GEM_ROOT)/cblas.h
$(GEM_OBJ): $(GEM_ROOT)/otlib/EMD_wrapper.cpp $(wildcard $(GEM_ROOT)/otlib/*.h) Makefile ../config.mk | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF $@.d -MT $@ -c $< -o $@

clean:
	$(RM) $(BINARIES) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
