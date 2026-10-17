#!/usr/bin/env python3
"""XTR/ScaNN token retrieval over the project's shared .mvbundle vectors."""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import scann


OFFICIAL_REPO = "https://github.com/google-deepmind/xtr"
OFFICIAL_COMMIT = "52d5b5ec796f51a8eb76aa727873545a58ce8b80"
HARNESS_REPO = "https://github.com/jlscheerer/xtr-eval"
HARNESS_COMMIT = "834a7793"


class NoopTracker:
    def begin(self, _name):
        pass

    def end(self, _name):
        pass


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(tmp), str(path))


def write_rankings(output_path, rankings, scores):
    output_path = Path(output_path)
    stem = output_path.with_suffix("") if output_path.suffix == ".json" else output_path
    data_path = Path(str(stem) + ".rankings.i32.bin")
    scores_path = Path(str(stem) + ".rankings.scores.fp32.bin")
    meta_path = Path(str(stem) + ".rankings.meta.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_tmp = data_path.with_name(data_path.name + ".tmp")
    np.ascontiguousarray(rankings, dtype=np.int32).tofile(data_tmp)
    os.replace(str(data_tmp), str(data_path))
    scores_tmp = scores_path.with_name(scores_path.name + ".tmp")
    np.ascontiguousarray(scores, dtype=np.float32).tofile(scores_tmp)
    os.replace(str(scores_tmp), str(scores_path))
    write_json(
        meta_path,
        {
            "format": "rankings_i32_f32_v1",
            "doc_ids_dtype": "int32",
            "scores_dtype": "float32",
            "shape": list(rankings.shape),
            "doc_id_padding_value": -1,
            "score_padding_value": "nan",
            "order": "method_final_rank",
            "doc_ids_path": str(data_path),
            "scores_path": str(scores_path),
            "score_semantics": "exact_maxsim_recomputed_after_search",
            "result_json": str(output_path),
        },
    )
    return data_path, scores_path, meta_path


def directory_size(path):
    logical = 0
    allocated = 0
    count = 0
    for entry in Path(path).rglob("*"):
        if not entry.is_file():
            continue
        stat = entry.stat()
        logical += stat.st_size
        allocated += getattr(stat, "st_blocks", 0) * 512
        count += 1
    return logical, allocated or logical, count


def bundle_shape(bundle, limit_docs=0):
    bundle = Path(bundle)
    meta = read_json(bundle / "meta.json")
    dim = int(meta["dim"])
    full_docs = int(meta["n_docs"])
    full_tokens = int(meta["n_tokens"])
    doclens = np.fromfile(bundle / "doclens.i32.bin", dtype=np.int32)
    if doclens.size != full_docs:
        raise ValueError("doclens count {} != {}".format(doclens.size, full_docs))
    n_docs = min(full_docs, limit_docs) if limit_docs else full_docs
    doclens = doclens[:n_docs]
    n_tokens = int(doclens.astype(np.int64).sum())
    if not limit_docs and n_tokens != full_tokens:
        raise ValueError("doclens token sum {} != {}".format(n_tokens, full_tokens))
    return meta, doclens, n_docs, n_tokens, dim


def load_embeddings(bundle, dtype, filename, n_tokens, dim):
    path = Path(bundle) / filename
    expected = n_tokens * dim * np.dtype(dtype).itemsize
    if path.stat().st_size < expected:
        raise ValueError("{} is smaller than expected {} bytes".format(path, expected))
    return np.memmap(path, mode="r", dtype=dtype, shape=(n_tokens, dim))


def load_embedding_range(bundle, dtype, filename, token_start, n_tokens, dim):
    path = Path(bundle) / filename
    itemsize = np.dtype(dtype).itemsize
    offset = token_start * dim * itemsize
    required = offset + n_tokens * dim * itemsize
    if path.stat().st_size < required:
        raise ValueError("{} is smaller than required {} bytes".format(path, required))
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=(n_tokens, dim),
    )


def distribute_total(total, count):
    return [total // count + (1 if i < total % count else 0) for i in range(count)]


def token_balanced_doc_boundaries(doclens, num_shards):
    offsets = np.empty(doclens.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(doclens.astype(np.int64), out=offsets[1:])
    boundaries = [0]
    for shard_id in range(1, num_shards):
        target = int(offsets[-1]) * shard_id // num_shards
        boundary = int(np.searchsorted(offsets, target, side="left"))
        boundary = max(boundaries[-1] + 1, min(boundary, doclens.size - (num_shards - shard_id)))
        boundaries.append(boundary)
    boundaries.append(int(doclens.size))
    return offsets, boundaries


def build(args):
    index_dir = Path(args.index_dir).resolve()
    if index_dir.exists():
        if not args.overwrite:
            raise FileExistsError("index already exists: {}".format(index_dir))
        shutil.rmtree(str(index_dir))
    index_dir.mkdir(parents=True)

    total_started = time.monotonic()
    bundle_meta, doclens, n_docs, n_tokens, dim = bundle_shape(
        args.doc_bundle, args.limit_docs
    )
    if bundle_meta.get("dtype") != "float16":
        raise ValueError("document bundle must contain float16 vectors")
    if dim != 128:
        raise ValueError("XTR ScaNN adapter expects dim=128, got {}".format(dim))
    print(
        "#> shared bundle docs={} tokens={} dim={} source={}".format(
            n_docs, n_tokens, dim, args.doc_bundle
        ),
        flush=True,
    )

    convert_started = time.monotonic()
    source = load_embeddings(
        args.doc_bundle, np.float16, "embeddings.fp16.bin", n_tokens, dim
    )
    if args.scann_input_dtype == "fp32":
        vectors = np.empty((n_tokens, dim), dtype=np.float32)
        np.copyto(vectors, source, casting="unsafe")
    else:
        vectors = source
    conversion_sec = time.monotonic() - convert_started
    print(
        "#> prepared ScaNN input dtype={} in {:.3f}s".format(
            args.scann_input_dtype, conversion_sec
        ),
        flush=True,
    )

    mapping_started = time.monotonic()
    token_docids = np.repeat(np.arange(n_docs, dtype=np.int32), doclens)
    if token_docids.size != n_tokens:
        raise AssertionError("token-to-document map size mismatch")
    doc_offsets = np.empty(n_docs + 1, dtype=np.int64)
    doc_offsets[0] = 0
    np.cumsum(doclens.astype(np.int64), out=doc_offsets[1:])
    mapping_sec = time.monotonic() - mapping_started

    build_started = time.monotonic()
    searcher = (
        scann.scann_ops_pybind.builder(vectors, 10, "dot_product")
        .set_n_training_threads(args.index_threads)
        .tree(
            num_leaves=min(2000, n_tokens),
            num_leaves_to_search=100,
            training_sample_size=min(250000, n_tokens),
        )
        .score_ah(1, anisotropic_quantization_threshold=0.1)
        .build()
    )
    scann_build_sec = time.monotonic() - build_started
    print("#> ScaNN/AH build completed in {:.3f}s".format(scann_build_sec), flush=True)

    save_started = time.monotonic()
    scann_dir = index_dir / "scann"
    scann_dir.mkdir()
    searcher.serialize(str(scann_dir))
    doclens.tofile(index_dir / "doclens.i32.bin")
    doc_offsets.tofile(index_dir / "doc_offsets.i64.bin")
    token_docids.tofile(index_dir / "token_docids.i32.bin")
    save_sec = time.monotonic() - save_started
    build_time_sec = time.monotonic() - total_started

    meta = {
        "method": "xtr_scann_mvbundle",
        "layout": "xtr_scann_ah_mvbundle_v1",
        "ann_backend": "scann_ah",
        "metric": "dot_product",
        "dataset": args.dataset,
        "split": args.split,
        "query_type_scope": args.query_type,
        "doc_bundle": str(Path(args.doc_bundle).resolve()),
        "embedding_source": "shared_mvbundle_embeddings",
        "shared_doc_store": "mvbundle",
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "dim": dim,
        "source_dtype": "float16",
        "scann_input_dtype": "float32" if args.scann_input_dtype == "fp32" else "float16",
        "scann_input_memory_mode": (
            "explicit_fp32_copy" if args.scann_input_dtype == "fp32" else "direct_fp16_memmap"
        ),
        "index_threads": args.index_threads,
        "conversion_sec": conversion_sec,
        "token_mapping_sec": mapping_sec,
        "scann_build_sec": scann_build_sec,
        "save_sec": save_sec,
        "build_time_sec": build_time_sec,
        "formal_baseline": "false" if args.limit_docs else "true",
        "parameter_status": "mvbundle_smoke" if args.limit_docs else "official_notebook_scann_default",
        "official_core_repo": OFFICIAL_REPO,
        "official_core_commit": OFFICIAL_COMMIT,
        "official_core_source": "xtr_evaluation_on_beir_miracl.ipynb",
        "execution_harness_repo": HARNESS_REPO,
        "execution_harness_commit": HARNESS_COMMIT,
        "scoring_kernel": "xtr_missing_similarity_and_mean_sum_of_max",
        "scann_num_neighbors": 10,
        "scann_max_num_leaves": 2000,
        "scann_num_leaves_to_search": 100,
        "scann_max_training_sample_size": 250000,
        "scann_dimensions_per_block": 1,
        "scann_anisotropic_quantization_threshold": 0.1,
        "search_leaves_to_search": 100,
        "search_pre_reorder_num_neighbors": 100,
        "python": platform.python_version(),
    }
    write_json(index_dir / "meta.json", meta)
    logical, allocated, count = directory_size(index_dir)
    meta.update(
        {
            "logical_file_bytes": logical,
            "allocated_disk_bytes": allocated,
            "file_count": count,
        }
    )
    write_json(index_dir / "meta.json", meta)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


def build_shard(args):
    index_dir = Path(args.index_dir).resolve()
    if index_dir.exists():
        if not args.overwrite:
            raise FileExistsError("index already exists: {}".format(index_dir))
        shutil.rmtree(str(index_dir))
    index_dir.mkdir(parents=True)

    total_started = time.monotonic()
    bundle_meta, full_doclens, full_docs, full_tokens, dim = bundle_shape(args.doc_bundle)
    if bundle_meta.get("dtype") != "float16" or dim != 128:
        raise ValueError("XTR shard build requires float16 dim=128 document vectors")
    if not (0 <= args.doc_start < args.doc_end <= full_docs):
        raise ValueError("invalid document range [{}, {})".format(args.doc_start, args.doc_end))

    full_offsets = np.empty(full_docs + 1, dtype=np.int64)
    full_offsets[0] = 0
    np.cumsum(full_doclens.astype(np.int64), out=full_offsets[1:])
    token_start = int(full_offsets[args.doc_start])
    token_end = int(full_offsets[args.doc_end])
    n_tokens = token_end - token_start
    doclens = full_doclens[args.doc_start : args.doc_end]
    n_docs = int(doclens.size)
    print(
        "#> shard docs=[{}, {}) tokens=[{}, {}) count={}".format(
            args.doc_start, args.doc_end, token_start, token_end, n_tokens
        ),
        flush=True,
    )

    convert_started = time.monotonic()
    source = load_embedding_range(
        args.doc_bundle,
        np.float16,
        "embeddings.fp16.bin",
        token_start,
        n_tokens,
        dim,
    )
    vectors = source
    conversion_sec = time.monotonic() - convert_started

    mapping_started = time.monotonic()
    token_docids = np.repeat(
        np.arange(args.doc_start, args.doc_end, dtype=np.int32), doclens
    )
    mapping_sec = time.monotonic() - mapping_started

    build_started = time.monotonic()
    searcher = (
        scann.scann_ops_pybind.builder(vectors, 10, "dot_product")
        .set_n_training_threads(args.index_threads)
        .tree(
            num_leaves=min(args.num_leaves, n_tokens),
            num_leaves_to_search=min(args.leaves_to_search, n_tokens),
            training_sample_size=min(args.training_sample_size, n_tokens),
        )
        .score_ah(1, anisotropic_quantization_threshold=0.1)
        .build()
    )
    scann_build_sec = time.monotonic() - build_started
    print("#> shard ScaNN/AH build completed in {:.3f}s".format(scann_build_sec), flush=True)

    save_started = time.monotonic()
    scann_dir = index_dir / "scann"
    scann_dir.mkdir()
    searcher.serialize(str(scann_dir))
    token_docids.tofile(index_dir / "token_docids.i32.bin")
    save_sec = time.monotonic() - save_started
    build_time_sec = time.monotonic() - total_started
    meta = {
        "method": "xtr_scann_mvbundle",
        "layout": "xtr_scann_ah_mvbundle_shard_v1",
        "ann_backend": "scann_ah",
        "metric": "dot_product",
        "dataset": args.dataset,
        "split": args.split,
        "doc_bundle": str(Path(args.doc_bundle).resolve()),
        "embedding_source": "shared_mvbundle_embeddings",
        "shared_doc_store": "mvbundle",
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "full_n_docs": full_docs,
        "full_n_tokens": full_tokens,
        "doc_id_start": args.doc_start,
        "doc_id_end": args.doc_end,
        "token_id_start": token_start,
        "token_id_end": token_end,
        "dim": dim,
        "source_dtype": "float16",
        "scann_input_dtype": "float16",
        "scann_input_memory_mode": "direct_fp16_memmap_range",
        "index_threads": args.index_threads,
        "conversion_sec": conversion_sec,
        "token_mapping_sec": mapping_sec,
        "scann_build_sec": scann_build_sec,
        "save_sec": save_sec,
        "build_time_sec": build_time_sec,
        "formal_baseline": "false",
        "component_status": "component_of_formal_sharded_index",
        "parameter_status": "official_totals_partitioned_for_bounded_memory",
        "official_core_repo": OFFICIAL_REPO,
        "official_core_commit": OFFICIAL_COMMIT,
        "official_core_source": "xtr_evaluation_on_beir_miracl.ipynb",
        "execution_harness_repo": HARNESS_REPO,
        "execution_harness_commit": HARNESS_COMMIT,
        "scann_num_neighbors": 10,
        "scann_num_leaves": args.num_leaves,
        "scann_num_leaves_to_search": args.leaves_to_search,
        "scann_training_sample_size": args.training_sample_size,
        "scann_dimensions_per_block": 1,
        "scann_anisotropic_quantization_threshold": 0.1,
        "python": platform.python_version(),
    }
    write_json(index_dir / "meta.json", meta)
    logical, allocated, count = directory_size(index_dir)
    meta.update({"logical_file_bytes": logical, "allocated_disk_bytes": allocated, "file_count": count})
    write_json(index_dir / "meta.json", meta)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


def build_sharded(args):
    index_dir = Path(args.index_dir).resolve()
    if index_dir.exists() and args.overwrite:
        shutil.rmtree(str(index_dir))
    index_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    bundle_meta, doclens, n_docs, n_tokens, dim = bundle_shape(args.doc_bundle)
    if bundle_meta.get("dtype") != "float16" or dim != 128:
        raise ValueError("XTR sharded build requires float16 dim=128 document vectors")
    offsets, boundaries = token_balanced_doc_boundaries(doclens, args.num_shards)
    leaf_counts = distribute_total(2000, args.num_shards)
    search_leaf_counts = distribute_total(100, args.num_shards)
    training_counts = distribute_total(250000, args.num_shards)
    shards_dir = index_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    shard_metas = []

    for shard_id in range(args.num_shards):
        shard_dir = shards_dir / "{:05d}".format(shard_id)
        expected = {
            "doc_id_start": boundaries[shard_id],
            "doc_id_end": boundaries[shard_id + 1],
            "scann_num_leaves": leaf_counts[shard_id],
            "scann_num_leaves_to_search": search_leaf_counts[shard_id],
            "scann_training_sample_size": training_counts[shard_id],
        }
        shard_meta_path = shard_dir / "meta.json"
        reusable = False
        if shard_meta_path.exists():
            current = read_json(shard_meta_path)
            reusable = all(current.get(key) == value for key, value in expected.items())
        if not reusable:
            if shard_dir.exists():
                shutil.rmtree(str(shard_dir))
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "build-shard",
                "--doc-bundle", args.doc_bundle,
                "--index-dir", str(shard_dir),
                "--dataset", args.dataset,
                "--split", args.split,
                "--index-threads", str(args.index_threads),
                "--doc-start", str(boundaries[shard_id]),
                "--doc-end", str(boundaries[shard_id + 1]),
                "--num-leaves", str(leaf_counts[shard_id]),
                "--leaves-to-search", str(search_leaf_counts[shard_id]),
                "--training-sample-size", str(training_counts[shard_id]),
            ]
            print("#> building shard {}/{}".format(shard_id + 1, args.num_shards), flush=True)
            subprocess.run(command, check=True)
        else:
            print("#> reusing completed shard {}".format(shard_id), flush=True)
        shard_metas.append(read_json(shard_meta_path))

    token_map_path = index_dir / "token_docids.i32.bin"
    with open(token_map_path.with_suffix(".tmp"), "wb") as output:
        for shard_id in range(args.num_shards):
            shard_map = shards_dir / "{:05d}".format(shard_id) / "token_docids.i32.bin"
            with open(shard_map, "rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    os.replace(str(token_map_path.with_suffix(".tmp")), str(token_map_path))
    if token_map_path.stat().st_size != n_tokens * np.dtype(np.int32).itemsize:
        raise ValueError("global token map size mismatch")
    for shard_id in range(args.num_shards):
        (shards_dir / "{:05d}".format(shard_id) / "token_docids.i32.bin").unlink()

    doc_offsets = np.empty(n_docs + 1, dtype=np.int64)
    doc_offsets[0] = 0
    np.cumsum(doclens.astype(np.int64), out=doc_offsets[1:])
    doclens.tofile(index_dir / "doclens.i32.bin")
    doc_offsets.tofile(index_dir / "doc_offsets.i64.bin")
    build_time_sec = time.monotonic() - started
    meta = {
        "method": "xtr_scann_mvbundle",
        "layout": "xtr_scann_ah_mvbundle_sharded_v1",
        "ann_backend": "scann_ah_sharded",
        "metric": "dot_product",
        "dataset": args.dataset,
        "split": args.split,
        "query_type_scope": args.query_type,
        "doc_bundle": str(Path(args.doc_bundle).resolve()),
        "embedding_source": "shared_mvbundle_embeddings",
        "shared_doc_store": "mvbundle",
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "dim": dim,
        "source_dtype": "float16",
        "scann_input_dtype": "float16",
        "scann_input_memory_mode": "serial_direct_fp16_memmap_shards",
        "index_threads": args.index_threads,
        "num_shards": args.num_shards,
        "shards": shard_metas,
        "scann_total_num_leaves": sum(leaf_counts),
        "scann_total_num_leaves_to_search": sum(search_leaf_counts),
        "scann_total_training_sample_size": sum(training_counts),
        "scann_dimensions_per_block": 1,
        "scann_anisotropic_quantization_threshold": 0.1,
        "build_time_sec": build_time_sec,
        "formal_baseline": "true",
        "parameter_status": "official_totals_partitioned_for_bounded_memory",
        "official_core_repo": OFFICIAL_REPO,
        "official_core_commit": OFFICIAL_COMMIT,
        "official_core_source": "xtr_evaluation_on_beir_miracl.ipynb",
        "execution_harness_repo": HARNESS_REPO,
        "execution_harness_commit": HARNESS_COMMIT,
        "scoring_kernel": "xtr_missing_similarity_and_mean_sum_of_max_after_global_token_topk_merge",
        "python": platform.python_version(),
    }
    write_json(index_dir / "meta.json", meta)
    logical, allocated, count = directory_size(index_dir)
    meta.update({"logical_file_bytes": logical, "allocated_disk_bytes": allocated, "file_count": count})
    write_json(index_dir / "meta.json", meta)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


def relocate_index(args):
    index_dir = Path(args.index_dir).resolve()
    old_root = str(Path(args.old_root).resolve())
    new_root = str(index_dir)
    manifests = sorted(index_dir.rglob("scann_assets.pbtxt"))
    if not manifests:
        raise FileNotFoundError("no ScaNN asset manifests under {}".format(index_dir))

    replaced = 0
    asset_paths = []
    for manifest in manifests:
        text = manifest.read_text(encoding="utf-8")
        updated = text.replace(old_root, new_root)
        if updated != text:
            tmp = manifest.with_name(manifest.name + ".tmp")
            tmp.write_text(updated, encoding="utf-8")
            os.replace(str(tmp), str(manifest))
            replaced += 1
        asset_paths.extend(re.findall(r'asset_path:\s*"([^"]+)"', updated))
    if replaced == 0:
        raise ValueError("old root not found in ScaNN asset manifests: {}".format(old_root))
    missing = [path for path in asset_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("relocated ScaNN assets missing: {}".format(missing))

    meta_path = index_dir / "meta.json"
    meta = read_json(meta_path)
    meta["relocated_from"] = old_root
    meta["relocated_to"] = new_root
    write_json(meta_path, meta)
    logical, allocated, count = directory_size(index_dir)
    meta.update({"logical_file_bytes": logical, "allocated_disk_bytes": allocated, "file_count": count})
    write_json(meta_path, meta)
    print(json.dumps({"status": "completed", "manifests": replaced, "assets": len(asset_paths), "index_dir": new_root}, indent=2), flush=True)


def load_gt(path, n_queries):
    path = Path(path)
    meta = read_json(path / "meta.json")
    top_k = int(meta["top_k"])
    values = np.fromfile(path / "topk_indices.i32.bin", dtype=np.int32)
    values = values.reshape(int(meta["n_queries"]), top_k)
    return values[:n_queries]


def recall_at_depth(approx, gt, k):
    if approx.shape[1] < k or gt.shape[1] < k:
        return 0.0
    return float(
        np.mean(
            [len(set(map(int, a[:k])).intersection(map(int, g[:k]))) / float(k) for a, g in zip(approx, gt)]
        )
    )


def mrr_at_depth(approx, gt, k):
    vals = []
    for row, truth in zip(approx, gt[:, 0]):
        ranks = np.flatnonzero(row[:k] == truth)
        vals.append(1.0 / float(ranks[0] + 1) if ranks.size else 0.0)
    return float(np.mean(vals))


def success_at_depth(approx, gt, k):
    return float(np.mean([int(truth) in set(map(int, row[:k])) for row, truth in zip(approx, gt[:, 0])]))


def exact_maxsim_scores(doc_bundle, query_vectors, query_offsets, rankings):
    _meta, doc_lens, n_docs, n_doc_tokens, dim = bundle_shape(doc_bundle)
    doc_vectors = load_embeddings(
        doc_bundle, np.float16, "embeddings.fp16.bin", n_doc_tokens, dim
    )
    doc_offsets = np.empty(n_docs + 1, dtype=np.int64)
    doc_offsets[0] = 0
    np.cumsum(doc_lens.astype(np.int64), out=doc_offsets[1:])
    scores = np.full(rankings.shape, np.nan, dtype=np.float32)
    for qi, row in enumerate(rankings):
        if qi % 100 == 0:
            print("#> export scores {}/{}".format(qi, rankings.shape[0]), flush=True)
        query = np.asarray(
            query_vectors[int(query_offsets[qi]) : int(query_offsets[qi + 1])],
            dtype=np.float32,
        )
        for rank, doc_id in enumerate(row):
            doc_id = int(doc_id)
            if doc_id < 0 or doc_id >= n_docs:
                continue
            doc = np.asarray(
                doc_vectors[int(doc_offsets[doc_id]) : int(doc_offsets[doc_id + 1])],
                dtype=np.float32,
            )
            scores[qi, rank] = float(np.max(query @ doc.T, axis=1).sum())
    return scores


def search(args):
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else {0}
    if args.search_threads != 1 or len(affinity) != 1:
        raise RuntimeError(
            "formal search requires search_threads=1 and one-CPU affinity; got threads={} affinity={}".format(
                args.search_threads, sorted(affinity)
            )
        )

    index_dir = Path(args.index_dir).resolve()
    index_meta = read_json(index_dir / "meta.json")
    if index_meta.get("embedding_source") != "shared_mvbundle_embeddings":
        raise ValueError("index does not use shared mvbundle embeddings")

    load_started = time.monotonic()
    query_meta, query_lens, n_queries, n_query_tokens, qdim = bundle_shape(
        args.query_bundle, args.limit_queries
    )
    if query_meta.get("dtype") != "float32":
        raise ValueError("query bundle must contain float32 vectors")
    if qdim != int(index_meta["dim"]):
        raise ValueError("query/index dimension mismatch")
    query_vectors = load_embeddings(
        args.query_bundle,
        np.float32,
        "embeddings.fp32.bin",
        n_query_tokens,
        qdim,
    )
    query_offsets = np.empty(n_queries + 1, dtype=np.int64)
    query_offsets[0] = 0
    np.cumsum(query_lens.astype(np.int64), out=query_offsets[1:])
    token_docids = np.memmap(
        index_dir / "token_docids.i32.bin",
        mode="r",
        dtype=np.int32,
        shape=(int(index_meta["n_tokens"]),),
    )
    sharded = index_meta.get("layout") == "xtr_scann_ah_mvbundle_sharded_v1"
    if sharded:
        shard_searchers = [
            (
                scann.scann_ops_pybind.load_searcher(
                    str(index_dir / "shards" / "{:05d}".format(shard_id) / "scann")
                ),
                shard_meta,
            )
            for shard_id, shard_meta in enumerate(index_meta["shards"])
        ]
        scann_searcher = None
    else:
        shard_searchers = []
        scann_searcher = scann.scann_ops_pybind.load_searcher(str(index_dir / "scann"))

    collection_paths = types.ModuleType("xtr.datasets.collection_paths")
    collection_paths.BEIR_COLLECTION_PATH = ""
    collection_paths.LOTTE_COLLECTION_PATH = ""
    sys.modules["xtr.datasets.collection_paths"] = collection_paths
    from xtr.modeling.xtr import XTR

    xtr_obj = XTR.__new__(XTR)
    xtr_obj.tid2did = token_docids
    tracker = NoopTracker()
    load_sec = time.monotonic() - load_started

    all_topk = np.full((n_queries, args.k), -1, dtype=np.int32)
    token_search_sec = 0.0
    aggregate_sec = 0.0
    search_started = time.monotonic()
    for qi in range(n_queries):
        if qi % 25 == 0:
            print("#> query {}/{}".format(qi, n_queries), flush=True)
        begin = int(query_offsets[qi])
        end = int(query_offsets[qi + 1])
        query = np.ascontiguousarray(query_vectors[begin:end], dtype=np.float32)
        t0 = time.monotonic()
        if sharded:
            shard_neighbors = []
            shard_scores = []
            for shard_searcher, shard_meta in shard_searchers:
                neighbors_part, scores_part = shard_searcher.search_batched(
                    query,
                    final_num_neighbors=args.token_top_k,
                    leaves_to_search=int(shard_meta["scann_num_leaves_to_search"]),
                    pre_reorder_num_neighbors=max(
                        1, int(shard_meta["scann_num_leaves_to_search"])
                    ),
                )
                shard_neighbors.append(
                    np.asarray(neighbors_part, dtype=np.int64)
                    + int(shard_meta["token_id_start"])
                )
                shard_scores.append(np.asarray(scores_part))
            merged_neighbors = np.concatenate(shard_neighbors, axis=1)
            merged_scores = np.concatenate(shard_scores, axis=1)
            keep = min(args.token_top_k, merged_scores.shape[1])
            selected = np.argpartition(merged_scores, -keep, axis=1)[:, -keep:]
            selected_scores = np.take_along_axis(merged_scores, selected, axis=1)
            order = np.argsort(selected_scores, axis=1)[:, ::-1]
            selected = np.take_along_axis(selected, order, axis=1)
            neighbors = np.take_along_axis(merged_neighbors, selected, axis=1)
            scores = np.take_along_axis(merged_scores, selected, axis=1)
        else:
            neighbors, scores = scann_searcher.search_batched(
                query,
                final_num_neighbors=args.token_top_k,
                leaves_to_search=100,
                pre_reorder_num_neighbors=100,
            )
        token_search_sec += time.monotonic() - t0
        result = [
            (
                ["q_{}".format(i) for i in range(query.shape[0])],
                np.asarray(neighbors),
                np.asarray(scores),
            )
        ]
        t1 = time.monotonic()
        missing = XTR._estimate_missing_similarity(xtr_obj, result, tracker)
        ranking = XTR._aggregate_scores(xtr_obj, result, missing, args.k, tracker)[0]
        aggregate_sec += time.monotonic() - t1
        for rank, (docid, _score) in enumerate(ranking[: args.k]):
            all_topk[qi, rank] = int(docid)

    search_total_sec = time.monotonic() - search_started
    all_scores = exact_maxsim_scores(
        args.doc_bundle, query_vectors, query_offsets, all_topk
    )
    rankings_data_path, rankings_scores_path, rankings_meta_path = write_rankings(
        args.output, all_topk, all_scores
    )
    metrics = {}
    if args.gt_dir:
        gt = load_gt(args.gt_dir, n_queries)
        metrics = {
            "recall@10": recall_at_depth(all_topk, gt, 10),
            "recall@100": recall_at_depth(all_topk, gt, 100),
            "recall@1000": recall_at_depth(all_topk, gt, 1000),
            "mrr@10": mrr_at_depth(all_topk, gt, 10),
            "success@5": success_at_depth(all_topk, gt, 5),
            "success@100": success_at_depth(all_topk, gt, 100),
        }
    output = {
        "status": "completed",
        "method": "xtr_scann_mvbundle",
        "dataset": args.dataset,
        "split": args.split,
        "query_type": args.query_type,
        "k": args.k,
        "params": {
            "formal_baseline": "false" if args.limit_queries else "true",
            "embedding_source": "shared_mvbundle_embeddings",
            "token_ann_backend": "scann_ah_sharded" if sharded else "scann_ah",
            "token_top_k": args.token_top_k,
            "leaves_to_search": index_meta.get("scann_total_num_leaves_to_search", 100),
            "pre_reorder_num_neighbors": index_meta.get("scann_total_num_leaves_to_search", 100),
            "scoring_kernel": "upstream_xtr_missing_similarity_and_aggregate_scores",
        },
        "metrics": metrics,
        "main_table_metrics": {
            "recall10@10": metrics.get("recall@10") if args.k == 10 else None,
            "recall100@100": metrics.get("recall@100") if args.k == 100 else None,
            "recall1000@1000": metrics.get("recall@1000") if args.k == 1000 else None,
        },
        "timing": {
            "build_sec": float(index_meta["build_time_sec"]),
            "load_sec": load_sec,
            "search_total_sec": search_total_sec,
            "search_per_query_ms": search_total_sec * 1000.0 / max(1, n_queries),
            "token_search_sec": token_search_sec,
            "aggregate_sec": aggregate_sec,
        },
        "runtime": {
            "search_threads": 1,
            "cpu_affinity": sorted(affinity),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        },
        "rankings": {
            "doc_ids_path": str(rankings_data_path),
            "scores_path": str(rankings_scores_path),
            "metadata_path": str(rankings_meta_path),
        },
        "index_size_bytes": directory_size(index_dir)[0],
        "n_queries": n_queries,
        "index_dir": str(index_dir),
        "query_bundle": str(Path(args.query_bundle).resolve()),
        "gt_dir": str(Path(args.gt_dir).resolve()) if args.gt_dir else "",
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--doc-bundle", required=True)
    build_parser.add_argument("--index-dir", required=True)
    build_parser.add_argument("--dataset", default="lotte_lifestyle")
    build_parser.add_argument("--split", default="test")
    build_parser.add_argument("--query-type", default="search")
    build_parser.add_argument("--index-threads", type=int, default=96)
    build_parser.add_argument(
        "--scann-input-dtype", choices=["fp32", "fp16"], default="fp32"
    )
    build_parser.add_argument("--limit-docs", type=int, default=0)
    build_parser.add_argument("--overwrite", action="store_true")

    shard_parser = subparsers.add_parser("build-shard")
    shard_parser.add_argument("--doc-bundle", required=True)
    shard_parser.add_argument("--index-dir", required=True)
    shard_parser.add_argument("--dataset", required=True)
    shard_parser.add_argument("--split", required=True)
    shard_parser.add_argument("--index-threads", type=int, default=16)
    shard_parser.add_argument("--doc-start", type=int, required=True)
    shard_parser.add_argument("--doc-end", type=int, required=True)
    shard_parser.add_argument("--num-leaves", type=int, required=True)
    shard_parser.add_argument("--leaves-to-search", type=int, required=True)
    shard_parser.add_argument("--training-sample-size", type=int, required=True)
    shard_parser.add_argument("--overwrite", action="store_true")

    sharded_parser = subparsers.add_parser("build-sharded")
    sharded_parser.add_argument("--doc-bundle", required=True)
    sharded_parser.add_argument("--index-dir", required=True)
    sharded_parser.add_argument("--dataset", required=True)
    sharded_parser.add_argument("--split", required=True)
    sharded_parser.add_argument("--query-type", default="search")
    sharded_parser.add_argument("--index-threads", type=int, default=16)
    sharded_parser.add_argument("--num-shards", type=int, default=3)
    sharded_parser.add_argument("--overwrite", action="store_true")

    relocate_parser = subparsers.add_parser("relocate-index")
    relocate_parser.add_argument("--index-dir", required=True)
    relocate_parser.add_argument("--old-root", required=True)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--index-dir", required=True)
    search_parser.add_argument("--doc-bundle", required=True)
    search_parser.add_argument("--query-bundle", required=True)
    search_parser.add_argument("--gt-dir")
    search_parser.add_argument("--output", required=True)
    search_parser.add_argument("--dataset", default="lotte_lifestyle")
    search_parser.add_argument("--split", default="test")
    search_parser.add_argument("--query-type", default="search")
    search_parser.add_argument("--k", type=int, required=True, choices=[10, 100, 1000])
    search_parser.add_argument("--token-top-k", type=int, required=True)
    search_parser.add_argument("--search-threads", type=int, default=1)
    search_parser.add_argument("--limit-queries", type=int, default=0)

    args = parser.parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "build-shard":
        build_shard(args)
    elif args.command == "build-sharded":
        build_sharded(args)
    elif args.command == "relocate-index":
        relocate_index(args)
    elif args.command == "search":
        search(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
