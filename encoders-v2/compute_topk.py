#!/usr/bin/env python3
"""
compute_topk.py — brute-force MaxSim ground truth for vector-search benchmarks.

For every (query, doc) pair compute:
    MaxSim(q, d) = sum_{q_t in q} max_{d_t in d} (q_t . d_t)
(token vectors are L2-normalized → q_t.d_t == cosine.)

Output layout (CLAUDE "scheme B": full scores + top-k convenience indices):

    <out>/
      scores.fp32.bin       (n_q, n_d) fp32 row-major — full MaxSim matrix
      topk_indices.i32.bin  (n_q, k) int32 — doc row indices, descending by score
      topk_scores.fp32.bin  (n_q, k) fp32 — matching scores (sanity / quick read)
      meta.json             {n_q, n_d, top_k, doc_bundle, query_bundle, sha256s,
                             score_dtype, sort_order, elapsed_sec, ...}

Reading later (numpy):
    scores = np.memmap(out/"scores.fp32.bin", dtype=np.float32, shape=(n_q, n_d))
    topk_i = np.fromfile(out/"topk_indices.i32.bin", dtype=np.int32).reshape(n_q, k)

Numerical recipe:
- doc embeddings on GPU as fp16 (matches ColBERT's serving dtype; see
  `colbert/modeling/colbert.py:175` — they cast Q to D's dtype, doing matmul
  in fp16).
- per-query q matrix cast to fp16 for the matmul.
- scatter_reduce(amax) and the final sum are done in fp32 for rigor (we are
  generating ground truth, not serving).
- scores written to disk in user-chosen dtype (default fp32).

Memory model (single chunk = whole doc bundle on GPU):
    GPU = doc fp16 (n_tok * dim * 2) + per-query scores fp16 (n_tok * q_maxlen * 2)
        + per-query scores fp32 (n_tok * q_maxlen * 4) [transient cast]
        + per-doc maxsim accumulator (n_docs * q_maxlen * 4)
    Worst case for lifestyle dev (40.76M tok, 269K docs, q_maxlen 32):
        9.8 GB + 2.6 GB + 5.2 GB + 0.03 GB ≈ 17.7 GB → fits 24 GB

If doc bundle exceeds GPU mem, pass --doc_chunk_tok to split by token range
(chunking implemented; per-query becomes n_chunks × work, but accumulates
correctly via per-chunk segment-reduce into the same maxsim accumulator row).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Reduce CUDA mem fragmentation when running near-OOM chunks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvbundle import MVBundle


MAGIC = "mvgt"
VERSION = 1


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--doc_bundle", required=True, help="doc .mvbundle directory")
    p.add_argument("--query_bundle", required=True, help="query .mvbundle directory")
    p.add_argument("--output", required=True, help="output .gt directory")
    p.add_argument("--top_k", type=int, default=10000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--score_dtype", default="fp32", choices=["fp32", "fp16"],
        help="dtype of stored scores.bin and topk_scores.bin",
    )
    p.add_argument(
        "--doc_chunk_tok", type=int, default=0,
        help="if >0, split doc bundle into chunks of this many tokens per GPU "
             "transfer. 0 = single chunk (require full bundle to fit on GPU).",
    )
    p.add_argument(
        "--query_start", type=int, default=0,
        help="start index in query bundle (inclusive). default 0.",
    )
    p.add_argument(
        "--query_stop", type=int, default=-1,
        help="stop index in query bundle (exclusive). -1 = all.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="smoke test: only first 10 queries × first 1000 docs, compare to "
             "naïve numpy implementation, exit after passing.",
    )
    p.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="force CPU for smoke test on machines without GPU.",
    )
    return p.parse_args()


def sha256_of(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def make_doc_id_per_tok(doclens: np.ndarray) -> np.ndarray:
    """Expand doclens [3,2,4] → [0,0,0, 1,1, 2,2,2,2]."""
    return np.repeat(np.arange(doclens.size, dtype=np.int64), doclens.astype(np.int64))


def maxsim_one_query(
    q_fp16: torch.Tensor,             # (q_maxlen, dim) fp16
    doc_emb_gpu: torch.Tensor,        # (n_tokens_in_chunk, dim) fp16
    doc_id_per_tok_gpu: torch.Tensor, # (n_tokens_in_chunk,) int64 — global doc id
    n_docs_full: int,
    accumulator: torch.Tensor,        # (n_docs_full, q_maxlen) fp16 or fp32, IN PLACE
) -> None:
    """Compute MaxSim contribution from this doc chunk into `accumulator`.

    Final MaxSim for doc d = accumulator[d].sum(dim=-1)  (callers do this).

    accumulator semantics: accumulator[d, t] = max over the doc-tokens seen so
    far of (q_token_t · d_token). For multi-chunk runs, call once per chunk
    with the same accumulator (initialized to -inf), and the running max is
    correct across chunks.

    Numerical strategy: matmul in fp16 (ColBERT-faithful); scatter_reduce_(amax)
    in accumulator's dtype. amax is arithmetic-free so fp16 amax is exact
    (no rounding error accumulation). Caller promotes to fp32 during the final
    sum to avoid summing 32 fp16 values with bounded precision.
    """
    q_maxlen, dim = q_fp16.shape
    scores_fp16 = doc_emb_gpu @ q_fp16.T  # (n_tok, q_maxlen) fp16
    # match accumulator dtype: fp16 source if accumulator is fp16, else cast
    src = scores_fp16 if accumulator.dtype == torch.float16 else scores_fp16.float()
    accumulator.scatter_reduce_(
        dim=0,
        index=doc_id_per_tok_gpu.unsqueeze(1).expand(-1, q_maxlen),
        src=src,
        reduce="amax",
        include_self=True,  # accumulator starts at -inf so include_self ok
    )


def smoke_test(db: MVBundle, qb: MVBundle, device: torch.device, top_k: int) -> None:
    """Compare GPU brute-force against naïve numpy on first 10 q × first 1000 docs."""
    n_q = min(10, qb.n_docs)
    n_d = min(1000, db.n_docs)
    q_maxlen = qb.meta["query_maxlen"]
    dim = db.dim
    print(f"[smoke] {n_q} queries × {n_d} docs, dim={dim}, q_maxlen={q_maxlen}")

    # Reference numpy (fp32 throughout)
    n_d_tok = int(db.doclens[:n_d].sum())
    doc_emb_np = np.asarray(db.embeddings[:n_d_tok], dtype=np.float32)
    doclens_np = db.doclens[:n_d]
    ref_maxsim = np.zeros((n_q, n_d), dtype=np.float32)
    for qi in range(n_q):
        q = np.asarray(qb[qi], dtype=np.float32)  # (q_maxlen, dim)
        scores = doc_emb_np @ q.T  # (n_d_tok, q_maxlen)
        off = 0
        for di in range(n_d):
            L = int(doclens_np[di])
            ref_maxsim[qi, di] = scores[off:off + L].max(axis=0).sum()
            off += L

    # GPU version (fp16 matmul + fp32 reduce)
    doc_emb_gpu = torch.from_numpy(doc_emb_np).half().to(device)
    doc_id_per_tok = torch.from_numpy(make_doc_id_per_tok(doclens_np)).to(device)
    got_maxsim = np.zeros((n_q, n_d), dtype=np.float32)
    for qi in range(n_q):
        q = torch.from_numpy(np.asarray(qb[qi], dtype=np.float32)).to(device)
        acc = torch.full((n_d, q_maxlen), -1e4, device=device, dtype=torch.float32)
        maxsim_one_query(q.half(), doc_emb_gpu, doc_id_per_tok, n_d, acc)
        got_maxsim[qi] = acc.sum(dim=1).cpu().numpy()

    diff = np.abs(ref_maxsim - got_maxsim)
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    # fp16 matmul error bound: ~5e-3 absolute per MaxSim score (32 tokens × 1e-4)
    tol = 5e-3
    print(f"[smoke] ref vs got: max_diff={max_d:.6f} mean_diff={mean_d:.6f} tol={tol}")
    assert max_d < tol, f"smoke test failed: max abs diff {max_d} > {tol}"

    # Also check top-k order agrees (allowing reorder of equal scores)
    k = min(top_k, n_d)
    ref_topk = np.argsort(-ref_maxsim, axis=1)[:, :k]
    got_topk = np.argsort(-got_maxsim, axis=1)[:, :k]
    # Loose: check top-10 overlap >= 8/10 across queries (fp16 noise can reorder)
    overlap_10 = np.mean([
        len(set(ref_topk[i, :10]) & set(got_topk[i, :10])) for i in range(n_q)
    ])
    print(f"[smoke] top-10 overlap (ref vs got): mean = {overlap_10:.2f} / 10")
    assert overlap_10 >= 8.0, f"top-10 overlap too low: {overlap_10}"
    print("[smoke] OK")


def main():
    args = parse_args()

    print(f"[load] doc bundle: {args.doc_bundle}")
    db = MVBundle.load(args.doc_bundle)
    if db.is_query:
        raise SystemExit("--doc_bundle must be a doc bundle (got is_query=True)")
    print(f"  {db}")

    print(f"[load] query bundle: {args.query_bundle}")
    qb = MVBundle.load(args.query_bundle)
    if not qb.is_query:
        raise SystemExit("--query_bundle must be a query bundle (got is_query=False)")
    if not qb.meta.get("keep_padding"):
        raise SystemExit(
            "--query_bundle was encoded without --keep_padding; this script "
            "currently assumes fixed-length queries. Re-encode with keep_padding=True."
        )
    print(f"  {qb}")

    if db.dim != qb.dim:
        raise SystemExit(f"dim mismatch: doc={db.dim} query={qb.dim}")

    dim = db.dim
    q_maxlen = qb.meta["query_maxlen"]
    n_docs = db.n_docs
    n_doc_tokens = db.n_tokens

    # query range
    q_start = args.query_start
    q_stop = qb.n_docs if args.query_stop < 0 else min(args.query_stop, qb.n_docs)
    if q_start >= q_stop:
        raise SystemExit(f"empty query range [{q_start}, {q_stop})")
    n_queries = q_stop - q_start
    top_k = min(args.top_k, n_docs)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("[warn] CUDA not available, falling back to CPU")
    if args.device == "cuda":
        device = torch.device(f"cuda:{args.gpu}")

    print(f"[plan] n_queries={n_queries} (rows [{q_start},{q_stop})) "
          f"n_docs={n_docs} dim={dim} q_maxlen={q_maxlen} top_k={top_k} device={device}")

    # ---- smoke ----
    if args.smoke:
        smoke_test(db, qb, device, top_k)
        return

    # ---- chunking plan ----
    # Auto-detect GPU mem and pick a chunk size if not specified.
    # Per-token live GPU cost (one query in-flight, fp16 path):
    #   doc fp16:              dim * 2          bytes (chunk-resident)
    #   per-query scores fp16: q_maxlen * 2     bytes (matmul output, transient)
    # Accumulator is fp16, n_docs × q_maxlen × 2 — usually negligible.
    if args.doc_chunk_tok == 0 and device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device.index or 0)
        safety = 4 * (1 << 30)  # framework slack + accumulator + topk buffer
        budget = max(1, free_bytes - safety)
        per_tok = dim * 2 + q_maxlen * 2  # 320 B at dim=128, q_maxlen=32
        auto_tok = int(budget // per_tok)
        if n_doc_tokens > auto_tok:
            args.doc_chunk_tok = auto_tok
            print(f"[auto-chunk] doc has {n_doc_tokens:,} tokens; GPU free "
                  f"{free_bytes / 1e9:.1f} GB, budget {budget / 1e9:.1f} GB, "
                  f"per-token {per_tok} B → chunking at {auto_tok:,} tokens/chunk")

    if args.doc_chunk_tok == 0 or args.doc_chunk_tok >= n_doc_tokens:
        # single chunk: load whole doc bundle on GPU
        chunks = [(0, n_docs)]  # by doc range
    else:
        # split by doc range so each chunk has ~args.doc_chunk_tok tokens
        chunks = []
        c0 = 0
        cur_tok = 0
        for i, L in enumerate(db.doclens):
            cur_tok += int(L)
            if cur_tok >= args.doc_chunk_tok:
                chunks.append((c0, i + 1))
                c0 = i + 1
                cur_tok = 0
        if c0 < n_docs:
            chunks.append((c0, n_docs))
        print(f"[chunk] {len(chunks)} chunks, sizes (docs): "
              f"{[e - s for s, e in chunks[:5]]}{'…' if len(chunks) > 5 else ''}")

    # ---- load queries (always small) ----
    q_all_np = np.asarray(qb.embeddings).reshape(qb.n_docs, q_maxlen, dim)  # fp32
    q_slice = q_all_np[q_start:q_stop]  # (n_q, q_maxlen, dim)
    q_gpu = torch.from_numpy(q_slice).to(device)  # fp32

    # ---- output ----
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_np_dtype = np.float32 if args.score_dtype == "fp32" else np.float16
    score_bytes = 4 if args.score_dtype == "fp32" else 2
    score_torch_dtype = torch.float32 if args.score_dtype == "fp32" else torch.float16

    scores_path = out_dir / f"scores.{args.score_dtype}.bin"
    topi_path = out_dir / "topk_indices.i32.bin"
    tops_path = out_dir / f"topk_scores.{args.score_dtype}.bin"
    meta_path = out_dir / "meta.json"

    expected_scores_bytes = n_queries * n_docs * score_bytes
    print(f"[write] scores → {scores_path} ({expected_scores_bytes / 1e9:.2f} GB)")
    print(f"[write] topk_indices → {topi_path}")
    print(f"[write] topk_scores → {tops_path}")

    # ---- per chunk: load doc to GPU, run all queries against it ----
    # accumulator persists across chunks (per query) so we use a 2-pass design:
    # 1. for each query: zero per-query accumulator; for each chunk update it
    # this requires n_chunks doc transfers PER query — wasteful.
    # better: outer loop chunks, inner loop queries; maintain (n_q, n_docs)
    # accumulator on GPU... but that's huge (7437 × 8.84M × q_maxlen × 4B = 8 TB!).
    #
    # so for single-chunk (the common case): outer loop is queries (simple).
    # for multi-chunk: outer loop chunks, inner loop queries; per-query maxsim
    # accumulator over n_docs × q_maxlen (~30 MB) is kept on CPU between chunks.

    t0 = time.time()
    last_log = t0

    if len(chunks) == 1:
        # ---- simple single-chunk path ----
        s, e = chunks[0]
        tok_s = int(db.offsets()[s])
        tok_e = int(db.offsets()[e])
        doc_emb_gpu = torch.from_numpy(
            np.asarray(db.embeddings[tok_s:tok_e])
        ).to(device)
        doc_id_per_tok = torch.from_numpy(
            make_doc_id_per_tok(db.doclens[s:e])
        ).to(device)
        print(f"[gpu] doc loaded: {doc_emb_gpu.shape} {doc_emb_gpu.dtype} "
              f"({doc_emb_gpu.numel() * 2 / 1e9:.2f} GB)")

        with open(scores_path, "wb") as f_scores, \
             open(topi_path,   "wb") as f_topi, \
             open(tops_path,   "wb") as f_tops:
            for qi in range(n_queries):
                q_fp16 = q_gpu[qi].half()
                # fp16 acc: amax is arithmetic-free so this is exact; promote
                # for the final sum to avoid 32-term fp16 summation error.
                acc = torch.full((n_docs, q_maxlen), -1e4,
                                 device=device, dtype=torch.float16)
                maxsim_one_query(q_fp16, doc_emb_gpu, doc_id_per_tok, n_docs, acc)
                maxsim = acc.sum(dim=1, dtype=torch.float32)  # (n_docs,) fp32

                # write full row
                row = maxsim.to(score_torch_dtype).cpu().numpy()
                row.tofile(f_scores)

                # top-k (descending)
                top_vals, top_idx = torch.topk(maxsim, top_k, largest=True, sorted=True)
                top_idx.to(torch.int32).cpu().numpy().tofile(f_topi)
                top_vals.to(score_torch_dtype).cpu().numpy().tofile(f_tops)

                now = time.time()
                if now - last_log >= 5.0 or qi + 1 == n_queries:
                    rate = (qi + 1) / max(now - t0, 1e-9)
                    eta = (n_queries - qi - 1) / max(rate, 1e-9)
                    print(f"[progress] {qi + 1}/{n_queries}  "
                          f"{rate:.1f} q/s  ETA {eta:.0f}s", flush=True)
                    last_log = now
    else:
        # ---- multi-chunk path: write scores into a mmap'd file as we go ----
        # Each doc range owns a contiguous *column* slice in scores.bin.
        # Pre-allocate scores.bin and write (q, chunk_doc_range) slices per
        # chunk × per query. No (n_q, n_d, q_maxlen) tensor needed.
        # Note: scores stored in `--score_dtype` (default fp32) for downstream
        # consistency with the single-chunk path.
        scores_mmap = np.memmap(scores_path, dtype=score_np_dtype, mode="w+",
                                shape=(n_queries, n_docs))
        print(f"[mmap] {scores_path} ({n_queries} × {n_docs} {score_np_dtype.__name__})")

        for ci, (s, e) in enumerate(chunks):
            tok_s = int(db.offsets()[s])
            tok_e = int(db.offsets()[e])
            chunk_n_docs = e - s
            print(f"[chunk {ci+1}/{len(chunks)}] docs [{s},{e}) "
                  f"= {chunk_n_docs} docs, {tok_e - tok_s} tokens "
                  f"({(tok_e - tok_s) * dim * 2 / 1e9:.1f} GB fp16)")
            doc_emb_gpu = torch.from_numpy(
                np.asarray(db.embeddings[tok_s:tok_e])
            ).to(device)
            doc_id_per_tok = torch.from_numpy(
                make_doc_id_per_tok(db.doclens[s:e])
            ).to(device)

            t_chunk = time.time()
            for qi in range(n_queries):
                q_fp16 = q_gpu[qi].half()
                acc = torch.full((chunk_n_docs, q_maxlen), -1e4,
                                 device=device, dtype=torch.float16)
                maxsim_one_query(q_fp16, doc_emb_gpu, doc_id_per_tok, chunk_n_docs, acc)
                # promote during sum, then cast to user score_dtype for storage
                maxsim_chunk = acc.sum(dim=1, dtype=torch.float32).to(score_torch_dtype).cpu().numpy()
                scores_mmap[qi, s:e] = maxsim_chunk

                now = time.time()
                if now - last_log >= 5.0 or qi + 1 == n_queries:
                    done = ci * n_queries + qi + 1
                    total = len(chunks) * n_queries
                    rate = (qi + 1) / max(now - t_chunk, 1e-9)
                    eta_chunk = (n_queries - qi - 1) / max(rate, 1e-9)
                    eta_total = (total - done) / max(done / (now - t0), 1e-9)
                    print(f"[chunk {ci+1}] q {qi+1}/{n_queries}  "
                          f"{rate:.1f} q/s  chunk_ETA {eta_chunk:.0f}s  "
                          f"total_ETA {eta_total:.0f}s", flush=True)
                    last_log = now
            del doc_emb_gpu, doc_id_per_tok
            torch.cuda.empty_cache() if device.type == "cuda" else None

        scores_mmap.flush()
        del scores_mmap  # close mmap so we can re-open read-only

        # ---- finalize: re-mmap scores, compute top-k per query ----
        print(f"[finalize] top-k over {n_queries} queries")
        scores_mmap_r = np.memmap(scores_path, dtype=score_np_dtype, mode="r",
                                  shape=(n_queries, n_docs))
        with open(topi_path, "wb") as f_topi, open(tops_path, "wb") as f_tops:
            for qi in range(n_queries):
                row = np.asarray(scores_mmap_r[qi], dtype=np.float32)
                if top_k < n_docs:
                    part = np.argpartition(-row, top_k - 1)[:top_k]
                    order = np.argsort(-row[part])
                    top_idx_np = part[order].astype(np.int32)
                else:
                    top_idx_np = np.argsort(-row)[:top_k].astype(np.int32)
                top_scores_np = row[top_idx_np].astype(score_np_dtype)
                top_idx_np.tofile(f_topi)
                top_scores_np.tofile(f_tops)

    elapsed = round(time.time() - t0, 2)
    print(f"[done] {n_queries} queries × {n_docs} docs in {elapsed}s")

    # ---- validate file sizes ----
    expect_topi = n_queries * top_k * 4
    expect_tops = n_queries * top_k * score_bytes
    actual_scores = scores_path.stat().st_size
    actual_topi = topi_path.stat().st_size
    actual_tops = tops_path.stat().st_size
    if actual_scores != expected_scores_bytes:
        raise SystemExit(
            f"scores.bin size mismatch: {actual_scores} vs expected {expected_scores_bytes}"
        )
    if actual_topi != expect_topi:
        raise SystemExit(
            f"topk_indices.bin size mismatch: {actual_topi} vs expected {expect_topi}"
        )
    if actual_tops != expect_tops:
        raise SystemExit(
            f"topk_scores.bin size mismatch: {actual_tops} vs expected {expect_tops}"
        )

    # ---- meta.json ----
    meta = {
        "magic": MAGIC,
        "version": VERSION,
        "n_queries": int(n_queries),
        "n_docs": int(n_docs),
        "top_k": int(top_k),
        "dim": int(dim),
        "q_maxlen": int(q_maxlen),
        "score_dtype": args.score_dtype,
        "sort_order": "descending",
        "query_range": [int(q_start), int(q_stop)],
        "doc_bundle": {
            "path": str(args.doc_bundle),
            "n_docs": int(db.n_docs),
            "n_tokens": int(db.n_tokens),
            "dtype": "float16",
            "meta_sha256": sha256_of(Path(args.doc_bundle) / "meta.json"),
        },
        "query_bundle": {
            "path": str(args.query_bundle),
            "n_queries": int(qb.n_docs),
            "n_tokens": int(qb.n_tokens),
            "dtype": "float32",
            "keep_padding": bool(qb.meta.get("keep_padding")),
            "meta_sha256": sha256_of(Path(args.query_bundle) / "meta.json"),
        },
        "compute": {
            "matmul_dtype": "float16",
            "reduce_dtype": "float32",
            "device": str(device),
            "chunks": len(chunks),
            "doc_chunk_tok": args.doc_chunk_tok,
            "elapsed_sec": elapsed,
        },
        "files": {
            scores_path.name: {"size": actual_scores,
                               "shape": [int(n_queries), int(n_docs)],
                               "dtype": args.score_dtype,
                               "sha256": sha256_of(scores_path)},
            topi_path.name:   {"size": actual_topi,
                               "shape": [int(n_queries), int(top_k)],
                               "dtype": "int32",
                               "sha256": sha256_of(topi_path)},
            tops_path.name:   {"size": actual_tops,
                               "shape": [int(n_queries), int(top_k)],
                               "dtype": args.score_dtype,
                               "sha256": sha256_of(tops_path)},
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] {meta_path}")


if __name__ == "__main__":
    main()
