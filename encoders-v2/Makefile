.DEFAULT_GOAL := all
include ../config.mk

BINARIES := $(BUILD_DIR)/mvbundle_inspect $(BUILD_DIR)/gt_inspect
.PHONY: all clean mvbundle.o mvbundle_inspect gt_inspect
all: $(BINARIES)
mvbundle.o: $(MVBUNDLE_OBJ)
mvbundle_inspect: $(BUILD_DIR)/mvbundle_inspect
gt_inspect: $(BUILD_DIR)/gt_inspect

$(BUILD_DIR)/mvbundle_inspect: $(BUILD_DIR)/mvbundle_inspect.o $(MVBUNDLE_OBJ)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD_DIR)/gt_inspect: $(BUILD_DIR)/gt_inspect.o $(MVBUNDLE_OBJ)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.cpp mvbundle.hpp Makefile ../config.mk | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF $@.d -MT $@ -c $< -o $@

clean:
	$(RM) $(BINARIES) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
