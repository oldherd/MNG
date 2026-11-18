# Data preparation and experiment guide

This guide documents the entry points in this repository. Commands are run from the repository root unless stated otherwise. Replace `/path/to/...` with your local paths. The [full paper](../fullpaper.pdf) describes the experimental settings; the commands below illustrate individual workflows and do not automatically reproduce every figure.

## Prepare the data

Download the LoTTE test collections and MS MARCO passage data through the links in the [README](../README.md#datasets). Obtain the ColBERTv2 checkpoint and install the dependencies of the [official ColBERT implementation](https://github.com/stanford-futuredata/ColBERT). The encoding script imports `colbert`, NumPy, and PyTorch and calls CUDA directly.

Prepare separate document and query TSV files, with one integer ID and one text field per line:

```text
0<TAB>First document or query text
1<TAB>Second document or query text
```

Here `<TAB>` denotes a literal tab. Keep a stable row order through encoding, index construction, ground-truth generation, and evaluation. For MS MARCO, the exact 500-query ID list used for the paper is not included; matching a random seed alone does not specify the sampling implementation or source ordering.

### Encode documents and queries

```bash
python encoders-v2/encode.py \
  --mode doc \
  --input /path/to/collection.tsv \
  --output data/docs.mvbundle \
  --ckpt /path/to/colbertv2-checkpoint \
  --vanilla /path/to/ColBERT \
  --dim 128 --doc_maxlen 220 --batch_size 64

python encoders-v2/encode.py \
  --mode query \
  --input /path/to/queries.tsv \
  --output data/queries.mvbundle \
  --ckpt /path/to/colbertv2-checkpoint \
  --vanilla /path/to/ColBERT \
  --dim 128 --query_maxlen 32 --batch_size 64 \
  --keep_padding
```

These are the encoder's default dimension and length settings, made explicit. `--vanilla` adds an existing ColBERT checkout to the Python import path; it can be omitted if `colbert` is already installed. `--dim` records the output dimension and must match the checkpoint. It does not retrain or resize its projection layer. Use `--limit N` for an initial small encoding run.

**Keep `--keep_padding` for queries:** the ground-truth generator requires fixed-length query bundles. The document bundle stores FP16 token vectors, while the query bundle stores FP32 token vectors.

```text
data/docs.mvbundle/          data/queries.mvbundle/
  embeddings.fp16.bin         embeddings.fp32.bin
  doclens.i32.bin             doclens.i32.bin
  docids.i32.bin              docids.i32.bin
  meta.json                  meta.json
```

`doclens.i32.bin` records the token count of each row. `docids.i32.bin` records its original input ID. Token vectors are concatenated in row order. MNG requires a document bundle with `is_query=false`, a query bundle with `is_query=true`, matching vector dimensions, and nonempty objects. The original document bundle is also required at search time.

Verify a generated bundle with the supplied reader:

```bash
python encoders-v2/mvbundle.py data/docs.mvbundle --verify
python encoders-v2/mvbundle.py data/queries.mvbundle --verify
```

### Generate MaxSim ground truth

```bash
python encoders-v2/compute_topk.py \
  --doc_bundle data/docs.mvbundle \
  --query_bundle data/queries.mvbundle \
  --output data/groundtruth.gt \
  --top_k 1000 --gpu 0
```

This enumerates query-document pairs using MaxSim. Token dot products use FP16 arithmetic and the final aggregation uses FP32. The output includes the full `scores.fp32.bin` matrix, `topk_indices.i32.bin`, `topk_scores.fp32.bin`, and `meta.json`. The full score matrix requires approximately `4 × number_of_queries × number_of_documents` bytes, in addition to the top-k files. Set `--top_k` at least as large as the largest search `--k` you intend to evaluate.

For GPU memory control, add `--doc_chunk_tok N` to process document tokens in chunks. A small numerical check is available with `--smoke`; add `--device cpu` to run that check without CUDA. Normal full ground-truth generation uses CUDA.

Ground truth must correspond to the same document and query bundles, including their row order and query count. The searcher does not align ground-truth rows by original query IDs. When using `--query_start` / `--query_stop`, prepare the matching query subset for evaluation as well.

## MNG search and evaluation

Follow the [build, index, and search commands](../README.md#compile-and-run-mng) in the README. For the data paths generated above, use `--doc data/docs.mvbundle`, `--query data/queries.mvbundle`, and `--gt data/groundtruth.gt`.

The parameter defaults below are taken from the argument parsers, rather than the example values in the usage message:

| Option | Code default | Meaning |
| --- | --- | --- |
| `--param M` | `32` | Routing Graph degree parameter |
| `--param M0` | `2*M` | Traversal Graph degree parameter |
| `--param assignment_gamma` | `M0` | Number of routing assignments used to gather construction candidates |
| `--param anchor_ratio` | `M` | Controls the size of the sampled routing subset |
| `--param ef_nav_build` | `128` | Routing Graph construction beam width |
| `--param ef_nav_assign` | `128` | Beam width for routing assignments during construction |
| `--param seed` | `42` | Construction random seed |
| `--param kmeans_niter` / `kmeans_nredo` | `25` / `5` | Local clustering iterations / restarts |
| `--param prox_builder` | `nearest_top_m0` | Traversal Graph construction variant |
| `--k` | `10` | Number of returned multi-vectors |
| `--param ef_nav_query` | `128` | Routing Graph search beam width (`efR` in the paper) |
| `--param ef_search` | `k` | Traversal Graph search beam width (`efS` in the paper) |
| `--param rerank_amp` | `0` | Exact MaxSim reranking amplification; `0` disables reranking |

Use `ef_search >= k > 0`. To enable exact MaxSim reranking, set `rerank_amp >= 1`. The reranked candidate count is bounded by `ceil(rerank_amp * k)`, `ef_search`, and the number of candidates found. The README explicitly uses `rerank_amp=2.64`, the value reported in the query-time ablation, as an example operating point. Increase or sweep `ef_search` to examine recall/latency trade-offs rather than assuming one setting reproduces every result.

The standard searcher requires a complete MNG index; it rejects the special `nav_only=1` construction mode.

### Output and metrics

For `--output results/mng.json`, the searcher writes:

- `results/mng.json`: timing, configuration, and evaluation summaries.
- `results/mng.rankings.i32.bin`: ranked document row indices.
- `results/mng.rankings.scores.fp32.bin`: corresponding scores.
- `results/mng.rankings.meta.json`: ranking dimensions and metadata.

Map ranking row indices to original document IDs using the document bundle's `docids.i32.bin`. Search without `--gt` is supported, but the uncomputed evaluation fields in the JSON must not be interpreted as measured recall values.

Search is forced to one software thread by the runtime. For controlled latency measurements, optionally pin the process to one available logical CPU using `taskset -c CPU_ID`; select an allowed CPU with `taskset -pc $$` on Linux. Keep the same affinity and evaluation settings across methods.

## Build options and baselines

`config.mk` provides the shared build settings:

| Variable | Purpose |
| --- | --- |
| `CXX`, `CXXFLAGS`, `CPPFLAGS`, `LDFLAGS`, `LDLIBS` | Compiler and additional compile/link flags |
| `FAISS_ROOT`, `FAISS_LIB_DIR`, `FAISS_LIBS` | FAISS installation and link configuration |
| `MKL_ROOT`, `MKL_LIB_DIR`, `MKL_LIBS` | Intel MKL installation and link configuration |
| `OPENBLAS_ROOT`, `OPENBLAS_LIB_DIR`, `OPENBLAS_LIBS` | OpenBLAS configuration for GEM |
| `EIGEN_INCLUDE_DIR` | Eigen headers for GEM; default `/usr/include/eigen3` |
| `MNG_BACKEND` | `auto`, `scalar`, `avx2`, or `avx512` |
| `MNG_INDEX_THREADS` | Main MNG construction threads; default `96` |

`MNG_BACKEND=auto` inspects the build machine's Linux CPU flags. If building for a different machine, select a backend supported by the execution CPU. The AVX-512 build uses AVX-512 F/DQ/VL and FMA; the AVX2 build uses AVX2 and FMA. The scalar setting affects MNG scoring kernels and does not remove the MKL/FAISS dependencies. The default MKL configuration links the GNU threading layer.

Make does not track changes to command-line variable values. After changing backend, threading, or library settings, rebuild the affected component:

```bash
make -C mng clean
make -C mng -j8 \
  FAISS_ROOT=/path/to/faiss MKL_ROOT=/path/to/mkl \
  MNG_BACKEND=avx2 MNG_INDEX_THREADS=16
```

The repository does not currently provide a locked dependency environment. The paper's recorded environment is listed in the README; other compiler and dependency combinations require validation on the target machine.

### Build targets

The root `make` target builds MNG, C++ bundle tools, all C++ baselines, and the ablation programs. Start with a selected component if you do not need every dependency:

```bash
make -j8 mng FAISS_ROOT=/path/to/faiss MKL_ROOT=/path/to/mkl
make -j8 encoders
make -C baselines -j8 indexer_dessert searcher_dessert
```

The C++ bundle tools are `encoders-v2/build/mvbundle_inspect` and `encoders-v2/build/gt_inspect`. Baseline binaries are placed under `baselines/build/`:

| Method | Build targets | Additional requirements |
| --- | --- | --- |
| ColBERT | `indexer_colbert`, `searcher_colbert` | FAISS, with the configured BLAS libraries |
| PLAID | `indexer_plaid`, `searcher_plaid` | FAISS for index construction |
| DESSERT | `indexer_dessert`, `searcher_dessert` | C++17 and OpenMP |
| WARP | `indexer_warp`, `searcher_warp` | FAISS for index construction |
| IGP | `indexer_igp`, `searcher_igp` | FAISS for index construction |
| MUVERA | `indexer_muvera_diskann`, `searcher_muvera_diskann` | Bundled ParlayANN/ParlayLib, jemalloc, and a precomputed FDE/PQ cache |
| GEM | `indexer_gem_official`, `searcher_gem_official` | Bundled GEM code, Eigen, OpenBLAS/gfortran, FAISS, and x86 AVX/FMA support |
| Brute-force MaxSim | `searcher_brute` | C++17 and OpenMP |

**Pass the paper's parameters explicitly.** Several baseline code defaults differ from the evaluation settings in Appendix A. For example, this DESSERT configuration explicitly selects the reported values:

```bash
mkdir -p indexes results
./baselines/build/indexer_dessert \
  --doc data/docs.mvbundle --index_dir indexes/dessert \
  --param L=32 --param C=7 --param n_filter=2000

./baselines/build/searcher_dessert \
  --doc data/docs.mvbundle --query data/queries.mvbundle \
  --gt data/groundtruth.gt --index_dir indexes/dessert --k 100 \
  --param filter_probe=4 --param filter_k=16384 \
  --output results/dessert.json
```

The MUVERA builder requires `--param source_index=/path/to/fde-cache`. It accepts existing `muvera_fde_hnsw` or `muvera_fde_cache` artifacts containing FDE/PQ data. A cache-generation entry point is not included in this repository, so this component is not a complete raw-data-to-index pipeline on its own.

### XTR

XTR uses `baselines/xtr/xtr_scann_mvbundle_runner.py`, separately from the C++ Make targets. It requires ScaNN and the dependencies imported by the bundled XTR package, including PyTorch, TensorFlow, FAISS Python bindings, Transformers, and its dataset/evaluation utilities. A pinned Python environment is not supplied.

```bash
python baselines/xtr/xtr_scann_mvbundle_runner.py build \
  --doc-bundle data/docs.mvbundle \
  --index-dir indexes/xtr \
  --dataset lotte_lifestyle --split test --index-threads 96

# Replace CPU_ID with an allowed logical CPU number.
taskset -c CPU_ID \
  python baselines/xtr/xtr_scann_mvbundle_runner.py search \
  --index-dir indexes/xtr \
  --doc-bundle data/docs.mvbundle \
  --query-bundle data/queries.mvbundle \
  --gt-dir data/groundtruth.gt \
  --dataset lotte_lifestyle --split test \
  --k 100 --token-top-k 40000 --search-threads 1 \
  --output results/xtr.json
```

The XTR runner checks single-CPU affinity as well as `--search-threads 1`; setting the thread argument alone is insufficient. Change the dataset/split labels to match your data.

### Ablations

| Directory | Binaries and purpose |
| --- | --- |
| `mng_brutemaxsim_ablation/` | `searcher_mng_brutemaxsim_ablation`: brute-force MaxSim during query processing, using a standard MNG index |
| `mng_build_maxsim_ablation/` | `indexer_mng_centroidmaxsim_build_control_optimized` and `indexer_mng_brutemaxsim_build_ablation`: accelerated versus exact MaxSim construction |
| `mng_build_maxsim_r95_search_quality/` | `searcher_mng_build_maxsim_r95_quality`: evaluates indexes from the preceding construction variants |

Build the ablation programs with `make ablations`, passing the same library paths as for MNG. The first two ablation Makefiles default `MNG_INDEX_THREADS` to `1`; explicitly pass `MNG_INDEX_THREADS=96` when that is the intended construction setting. The construction-quality searcher accepts its matching ablation index formats, rather than the standard MNG index format.
