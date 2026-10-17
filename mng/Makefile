.DEFAULT_GOAL := all
include ../config.mk

INCLUDES := -I. -I.. -I$(MVBUNDLE_DIR)
HEADERS := $(wildcard *.hpp ../common/*.hpp)
BIN_NAMES := indexer_mng_exp searcher_mng_exp
OPTIONAL_BIN_NAMES := searcher_mng_qid_exp
BINARIES := $(addprefix $(BUILD_DIR)/,$(BIN_NAMES) $(OPTIONAL_BIN_NAMES))
SEARCHERS := $(addprefix $(BUILD_DIR)/,searcher_mng_exp searcher_mng_qid_exp)
MNG_BUILD_FLAGS := $(BACKEND_FLAGS) $(MNG_FLAGS) -DMNG_FIXED_TARGET_REVERSE_BATCH=0

.PHONY: all clean $(BIN_NAMES) $(OPTIONAL_BIN_NAMES)
all: $(addprefix $(BUILD_DIR)/,$(BIN_NAMES))

$(SEARCHERS): EXTRA_CPPFLAGS += $(MKL_CPPFLAGS)
$(SEARCHERS): EXTRA_CXXFLAGS += $(RERANK_FLAGS)
$(BUILD_DIR)/searcher_mng_qid_exp: EXTRA_CXXFLAGS += -DMNG_QUERY_SEED_EXTERNAL_ID=1

# Put source prerequisites first: $< must remain the entrypoint source.
define binary
$(BUILD_DIR)/$(1): $(2) $(HEADERS) $(MVBUNDLE_OBJ) Makefile ../config.mk | $(BUILD_DIR)
	$$(CXX) $$(CPPFLAGS) $$(FAISS_CPPFLAGS) $$(EXTRA_CPPFLAGS) $$(INCLUDES) $$(CXXFLAGS) $$(OPENMP_FLAGS) $$(MNG_BUILD_FLAGS) $$(EXTRA_CXXFLAGS) -MMD -MP -MF $$@.d -MT $$@ $$< $$(MVBUNDLE_OBJ) $$(LDFLAGS) $$(FAISS_LDFLAGS) $$(MKL_LDFLAGS) $$(FAISS_LIBS) $$(LDLIBS) -o $$@
$(1): $(BUILD_DIR)/$(1)
endef
$(eval $(call binary,indexer_mng_exp,mng_indexer.cpp))
$(eval $(call binary,searcher_mng_exp,mng_searcher.cpp))
$(eval $(call binary,searcher_mng_qid_exp,mng_searcher.cpp))

clean:
	$(RM) $(BINARIES) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
