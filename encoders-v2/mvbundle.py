"""
mvbundle.py — read interface for multi-vector bundles produced by encode.py.

A bundle directory:
  embeddings.fp16.bin   doc mode: fp16 (N_tok, dim)
  embeddings.fp32.bin   query mode: fp32 (N_tok, dim)
  doclens.i32.bin       int32 (N_rows,) token count per row
  docids.i32.bin        int32 (N_rows,) original id per row (doc id / query id)
  meta.json             dtype, dim, counts, sha256, encode args

The embedding file is mmap'd, so loading a 100 GB bundle is O(1) and accesses
are paged on demand. Per-row slicing is constant time via a cached cumulative
offset table.

Doc bundle: each row is one document, variable token count.
Query bundle: each row is one query. When encoded with --keep_padding, every
row has exactly query_maxlen tokens (= 32 by default); without --keep_padding,
rows are variable-length.

Usage
-----
    from mvbundle import MVBundle

    b = MVBundle.load("/path/to/lotte_lifestyle_dev.mvbundle")
    b.n_docs, b.n_tokens, b.dim, b.dtype, b.is_query

    vecs = b[42]                  # (doclens[42], 128) view, fp16 for docs
    vecs = b.get_by_id(12345)     # lookup by original doc/query id

    for i, vecs in b.iter():
        ...                       # vecs is a (Lᵢ, dim) ndarray view

    # numpy memmap'd arrays — touch only if you need raw access
    b.embeddings, b.doclens, b.docids
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

# Avoid PEP 604 `str | os.PathLike` syntax (Python 3.10+); use Union alias
# instead, so this module works on the system Python 3.6 on the dev box.
PathLike = Union[str, os.PathLike]


MAGIC = "mvbin"
SUPPORTED_VERSIONS = {1}

_DTYPE_TO_NP = {
    "float16": np.float16,
    "float32": np.float32,
}
_DTYPE_TO_BYTES = {np.float16: 2, np.float32: 4}


@dataclass
class MVBundle:
    """Read-only view into a multi-vector bundle directory."""

    path: Path
    meta: dict
    dim: int
    dtype: np.dtype
    n_docs: int
    n_tokens: int
    is_query: bool
    embeddings: np.memmap         # (n_tokens, dim)
    doclens: np.ndarray           # (n_docs,) int32
    docids: np.ndarray            # (n_docs,) int32
    _offsets: np.ndarray          # (n_docs + 1,) int64 cumulative starts
    _id_to_row: Optional[dict]    # lazy

    # ---- construction -----------------------------------------------------

    @classmethod
    def load(cls, path: PathLike) -> "MVBundle":
        path = Path(path)
        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {path}")
        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("magic") != MAGIC:
            raise ValueError(f"bad magic in {meta_path}: {meta.get('magic')!r}")
        if meta.get("version") not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported version {meta.get('version')} (supported: {SUPPORTED_VERSIONS})"
            )

        dtype_name = meta["dtype"]
        if dtype_name not in _DTYPE_TO_NP:
            raise ValueError(f"unknown dtype in meta: {dtype_name!r}")
        np_dtype = _DTYPE_TO_NP[dtype_name]
        bytes_per_val = _DTYPE_TO_BYTES[np_dtype]

        dim = int(meta["dim"])
        n_docs = int(meta["n_docs"])
        n_tokens = int(meta["n_tokens"])
        is_query = bool(meta.get("is_query", False))

        emb_fname = "embeddings.fp32.bin" if is_query else "embeddings.fp16.bin"
        emb_path = path / emb_fname
        if not emb_path.exists():
            # fall back to the other extension if meta disagrees with file layout
            other = "embeddings.fp16.bin" if is_query else "embeddings.fp32.bin"
            if (path / other).exists():
                raise ValueError(
                    f"meta says dtype={dtype_name} (is_query={is_query}) but found "
                    f"{other} instead of {emb_fname} in {path}"
                )
            raise FileNotFoundError(f"{emb_fname} not found in {path}")

        expected_size = n_tokens * dim * bytes_per_val
        actual_size = emb_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"size mismatch for {emb_path.name}: "
                f"got {actual_size} bytes, expected {expected_size} "
                f"(n_tokens={n_tokens} × dim={dim} × {bytes_per_val})"
            )

        # mmap'd embeddings, paged in on demand
        emb = np.memmap(emb_path, dtype=np_dtype, mode="r", shape=(n_tokens, dim))

        doclens = np.fromfile(path / "doclens.i32.bin", dtype=np.int32)
        docids = np.fromfile(path / "docids.i32.bin", dtype=np.int32)
        if doclens.shape != (n_docs,):
            raise ValueError(f"doclens shape {doclens.shape} ≠ ({n_docs},)")
        if docids.shape != (n_docs,):
            raise ValueError(f"docids shape {docids.shape} ≠ ({n_docs},)")
        if int(doclens.sum()) != n_tokens:
            raise ValueError(
                f"doclens.sum() = {int(doclens.sum())} ≠ n_tokens = {n_tokens}"
            )

        offsets = np.empty(n_docs + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(doclens.astype(np.int64), out=offsets[1:])

        return cls(
            path=path,
            meta=meta,
            dim=dim,
            dtype=np_dtype,
            n_docs=n_docs,
            n_tokens=n_tokens,
            is_query=is_query,
            embeddings=emb,
            doclens=doclens,
            docids=docids,
            _offsets=offsets,
            _id_to_row=None,
        )

    # ---- core accessors ---------------------------------------------------

    def __len__(self) -> int:
        return self.n_docs

    def __getitem__(self, row: int) -> np.ndarray:
        """Return token vectors for the row-th item, shape (doclens[row], dim)."""
        if row < 0:
            row += self.n_docs
        if not 0 <= row < self.n_docs:
            raise IndexError(row)
        s = int(self._offsets[row])
        e = int(self._offsets[row + 1])
        return self.embeddings[s:e]

    def get_by_id(self, original_id: int) -> np.ndarray:
        """Look up by original doc/query id (the value stored in docids.i32.bin)."""
        if self._id_to_row is None:
            self._id_to_row = {int(i): r for r, i in enumerate(self.docids)}
        row = self._id_to_row.get(int(original_id))
        if row is None:
            raise KeyError(f"id {original_id} not in bundle")
        return self[row]

    def offsets(self) -> np.ndarray:
        """(n_docs + 1,) int64 cumulative token offsets; offsets[i+1]-offsets[i] = doclens[i]."""
        return self._offsets

    # ---- iteration --------------------------------------------------------

    def iter(self, start: int = 0, stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (row_idx, token_vectors) for rows in [start, stop)."""
        if stop is None:
            stop = self.n_docs
        for r in range(start, stop):
            yield r, self[r]

    def iter_with_ids(self, start: int = 0, stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Like iter(), but yields (original_id, token_vectors)."""
        if stop is None:
            stop = self.n_docs
        for r in range(start, stop):
            yield int(self.docids[r]), self[r]

    # ---- batched access (preferred for ML pipelines) ----------------------

    def gather(self, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate token vectors for the given rows.

        Returns (concat_vectors, lens) where lens[i] = doclens[rows[i]] and
        concat_vectors has shape (lens.sum(), dim). Useful for batched MaxSim.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.empty((0, self.dim), dtype=self.dtype), np.empty(0, dtype=np.int32)
        lens = self.doclens[rows].astype(np.int32)
        total = int(lens.sum())
        out = np.empty((total, self.dim), dtype=self.dtype)
        cursor = 0
        for r, L in zip(rows.tolist(), lens.tolist()):
            s = int(self._offsets[r])
            out[cursor:cursor + L] = self.embeddings[s:s + L]
            cursor += L
        return out, lens

    # ---- integrity --------------------------------------------------------

    def verify_checksums(self) -> dict:
        """Recompute sha256 of all listed files and return a {name: ok} map.

        Slow (re-reads the whole bundle). Call once after rsync to confirm.
        """
        result = {}
        for fname, info in self.meta.get("files", {}).items():
            fpath = self.path / fname
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            got = h.hexdigest()
            expected = info["sha256"]
            result[fname] = (got == expected)
        return result

    # ---- repr -------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "query" if self.is_query else "doc"
        return (
            f"MVBundle({kind}, {self.n_docs:,} rows, {self.n_tokens:,} tokens, "
            f"dim={self.dim}, dtype={self.dtype.__name__}, path={self.path.name})"
        )


# Module-level convenience.
def load(path: PathLike) -> MVBundle:
    """Shorthand for MVBundle.load(path)."""
    return MVBundle.load(path)


# ============================================================
# Ground truth (output of compute_topk.py)
# ============================================================

MAGIC_GT = "mvgt"
SUPPORTED_GT_VERSIONS = {1}


@dataclass
class MVGroundTruth:
    """Read-only view of a ground-truth directory produced by compute_topk.py.

    Layout:
        <gt_dir>/
          scores.{fp32|fp16}.bin    (n_q, n_d) row-major, full MaxSim matrix (mmap'd)
          topk_indices.i32.bin      (n_q, top_k) descending — loaded fully into RAM
          topk_scores.{fp32|fp16}.bin (n_q, top_k) matching scores
          meta.json                 magic="mvgt" v1

    Use:
        gt = MVGroundTruth.load("…/lotte_lifestyle_dev_search.gt")
        gt.n_queries, gt.n_docs, gt.top_k
        gt.scores                  # (n_q, n_d) memmap; gt.scores[q, d] = MaxSim
        gt.topk_indices            # (n_q, top_k) int32 in-RAM
        gt.topk_scores             # (n_q, top_k) same dtype as scores
        gt.recall_at_k(approx_topk, q, k)   # |approx_topk[:k] ∩ gt[q,:k]| / k
    """

    path: Path
    meta: dict
    n_queries: int
    n_docs: int
    top_k: int
    dim: int
    q_maxlen: int
    score_dtype: np.dtype
    scores: np.memmap                 # (n_q, n_d) score_dtype
    topk_indices: np.ndarray          # (n_q, top_k) int32
    topk_scores: np.ndarray           # (n_q, top_k) score_dtype

    @classmethod
    def load(cls, path: PathLike) -> "MVGroundTruth":
        path = Path(path)
        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"gt meta.json not found in {path}")
        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("magic") != MAGIC_GT:
            raise ValueError(f"bad gt magic: {meta.get('magic')!r}")
        if meta.get("version") not in SUPPORTED_GT_VERSIONS:
            raise ValueError(f"unsupported gt version {meta.get('version')}")

        n_q = int(meta["n_queries"])
        n_d = int(meta["n_docs"])
        top_k = int(meta["top_k"])
        dim = int(meta.get("dim", 0))
        q_maxlen = int(meta.get("q_maxlen", 0))

        sd = meta["score_dtype"]
        if sd in ("fp32", "float32"):
            np_score_dtype = np.float32
        elif sd in ("fp16", "float16"):
            np_score_dtype = np.float16
        else:
            raise ValueError(f"unknown score_dtype {sd!r}")
        bytes_per_score = 4 if np_score_dtype == np.float32 else 2

        scores_fname = "scores.fp32.bin" if np_score_dtype == np.float32 else "scores.fp16.bin"
        tops_fname   = "topk_scores.fp32.bin" if np_score_dtype == np.float32 else "topk_scores.fp16.bin"

        scores_path = path / scores_fname
        tops_path   = path / tops_fname
        topi_path   = path / "topk_indices.i32.bin"

        for p in (scores_path, tops_path, topi_path):
            if not p.exists():
                raise FileNotFoundError(f"{p.name} not found in {path}")

        # size checks
        expected_scores = n_q * n_d * bytes_per_score
        if scores_path.stat().st_size != expected_scores:
            raise ValueError(
                f"{scores_fname} size mismatch: got {scores_path.stat().st_size}, "
                f"expected {expected_scores}"
            )
        expected_tops = n_q * top_k * bytes_per_score
        if tops_path.stat().st_size != expected_tops:
            raise ValueError(f"{tops_fname} size mismatch")
        if topi_path.stat().st_size != n_q * top_k * 4:
            raise ValueError("topk_indices.i32.bin size mismatch")

        # mmap scores; load topk_* fully into RAM
        scores = np.memmap(scores_path, dtype=np_score_dtype, mode="r", shape=(n_q, n_d))
        topi   = np.fromfile(topi_path, dtype=np.int32).reshape(n_q, top_k)
        tops   = np.fromfile(tops_path, dtype=np_score_dtype).reshape(n_q, top_k)

        return cls(
            path=path,
            meta=meta,
            n_queries=n_q,
            n_docs=n_d,
            top_k=top_k,
            dim=dim,
            q_maxlen=q_maxlen,
            score_dtype=np_score_dtype,
            scores=scores,
            topk_indices=topi,
            topk_scores=tops,
        )

    def __len__(self) -> int:
        return self.n_queries

    def scores_row(self, q: int) -> np.ndarray:
        """Per-query full score row (n_docs,), zero-copy mmap view."""
        return self.scores[q]

    def gt_topk(self, q: int, k: Optional[int] = None) -> np.ndarray:
        """GT doc ids for query q, top-k (descending). Default k = stored top_k."""
        if k is None:
            k = self.top_k
        if k > self.top_k:
            raise ValueError(f"k={k} > stored top_k={self.top_k}")
        return self.topk_indices[q, :k]

    def recall_at_k(self, approx_topk: np.ndarray, q: int, k: int) -> float:
        """|approx_topk[:k] ∩ gt[q, :k]| / k."""
        if k > self.top_k:
            raise ValueError(f"k={k} > stored top_k={self.top_k}")
        return recall_at_k(approx_topk, self.topk_indices[q], k)

    def __repr__(self) -> str:
        return (
            f"MVGroundTruth({self.n_queries:,} q × {self.n_docs:,} d, "
            f"top_k={self.top_k}, dtype={self.score_dtype.__name__}, "
            f"path={self.path.name})"
        )


def recall_at_k(approx_topk: np.ndarray, gt_topk: np.ndarray, k: int) -> float:
    """Recall@k = |approx_topk[:k] ∩ gt_topk[:k]| / k."""
    if k == 0:
        return 0.0
    a = set(int(x) for x in np.asarray(approx_topk).flat[:k])
    g = np.asarray(gt_topk).flat[:k]
    hits = sum(1 for x in g if int(x) in a)
    return hits / k


def load_gt(path: PathLike) -> MVGroundTruth:
    """Shorthand for MVGroundTruth.load(path)."""
    return MVGroundTruth.load(path)


# ---- CLI smoke test -------------------------------------------------------

def _inspect_gt(path: str, show: int) -> None:
    gt = load_gt(path)
    print(gt)
    m = gt.meta
    print(f"  dim={gt.dim} q_maxlen={gt.q_maxlen} score_dtype={gt.score_dtype.__name__}")
    print(f"  sort_order={m.get('sort_order')} elapsed_sec={m.get('compute', {}).get('elapsed_sec')}")
    if "doc_bundle" in m:
        print(f"  doc_bundle:   {m['doc_bundle'].get('path')}")
    if "query_bundle" in m:
        print(f"  query_bundle: {m['query_bundle'].get('path')}")
    # show top scores for first few queries
    for q in range(min(show, gt.n_queries)):
        idx = gt.topk_indices[q]
        sc  = gt.topk_scores[q]
        print(f"  q[{q}]: top-3 = "
              f"{[(int(idx[i]), float(sc[i])) for i in range(min(3, gt.top_k))]}")
    # sanity: scores[q, topk_indices[q, 0]] should equal topk_scores[q, 0]
    q = 0
    s_from_full = float(gt.scores[q, int(gt.topk_indices[q, 0])])
    s_from_topk = float(gt.topk_scores[q, 0])
    diff = abs(s_from_full - s_from_topk)
    print(f"  consistency check (q=0, rank 0): full[q, gt_id] = {s_from_full:.6f}  "
          f"vs topk_scores = {s_from_topk:.6f}  diff = {diff:.6g}")


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="mvbundle / gt inspector")
    ap.add_argument("path", help="bundle directory (.mvbundle or .gt)")
    ap.add_argument("--show", type=int, default=3, help="print first N rows summary")
    ap.add_argument("--verify", action="store_true", help="recompute sha256 (slow)")
    ap.add_argument("--gt", action="store_true", help="treat path as a .gt directory")
    args = ap.parse_args()

    if args.gt or (Path(args.path) / "topk_indices.i32.bin").exists():
        _inspect_gt(args.path, args.show)
        return

    b = load(args.path)
    print(b)
    print(f"  doc_maxlen={b.meta.get('doc_maxlen')} query_maxlen={b.meta.get('query_maxlen')}")
    print(f"  keep_padding={b.meta.get('keep_padding')} encoder_ckpt={b.meta.get('encoder_ckpt')}")
    print(f"  doclens: min={int(b.doclens.min())} max={int(b.doclens.max())} "
          f"mean={float(b.doclens.mean()):.1f} sum={int(b.doclens.sum()):,}")
    print(f"  docids[:5]={b.docids[:5].tolist()} docids[-5:]={b.docids[-5:].tolist()}")

    for r in range(min(args.show, b.n_docs)):
        v = b[r]
        norm = float(np.linalg.norm(v[0].astype(np.float32)))
        print(f"  row {r}: id={int(b.docids[r])} len={v.shape[0]} dtype={v.dtype} "
              f"first_tok_L2={norm:.6f}")

    if args.verify:
        print("verifying sha256 ...")
        ok = b.verify_checksums()
        for fname, good in ok.items():
            print(f"  {fname}: {'OK' if good else 'MISMATCH'}")


if __name__ == "__main__":
    _main()
