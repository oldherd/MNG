.DEFAULT_GOAL := all
COMPONENTS := mng encoders-v2 baselines mng_brutemaxsim_ablation \
	mng_build_maxsim_ablation mng_build_maxsim_r95_search_quality

.PHONY: all encoders ablations clean $(COMPONENTS)
all: $(COMPONENTS)
encoders: encoders-v2
ablations: mng_brutemaxsim_ablation mng_build_maxsim_ablation mng_build_maxsim_r95_search_quality

$(COMPONENTS):
	$(MAKE) -C $@

clean:
	@for dir in $(COMPONENTS); do $(MAKE) -C "$$dir" clean || exit $$?; done
