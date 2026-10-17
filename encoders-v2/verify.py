#!/usr/bin/env python3
"""
verify.py — Compare a new fp16 SoA bundle against the legacy fp32 .fivecs
on the *first N docs*, computing per-token cosine similarity.

Both old and new vectors are expected to be L2-normalized (ColBERT does this
internally), so cosine == dot product. We expect cos > 0.999 for every token
when the only difference is fp32 → fp16 cast.

Mismatches typically come from:
  - different doc_maxlen (old fivecs may have used 300, new may have used 220)
  - different ColBERT codebase (user's modified ColBERT vs vanilla)
  - different tokenizer config
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_old_fivecs_until_did(path, max_did):
    """Stream old fp32 fivecs; collect every record with did < max_did. Returns dict did → list[fp32 vec]."""
    with open(path, "rb") as f:
        header = f.read(12)
        total_vectors, total_docs, dim = struct.unpack("<3i", header)
        record_size = 8 + 4 * dim
        bufsize = 100000  # records per read
        out = {}
        seen_done = False
        while not seen_done:
            chunk = f.read(record_size * bufsize)
            if not chunk:
                break
            n = len(chunk) // record_size
            for i in range(n):
                base = i * record_size
                vid, did = struct.unpack_from("<2i", chunk, base)
                if did >= max_did:
                    seen_done = True
                    break
                vec = np.frombuffer(chunk, dtype=np.float32, count=dim,
                                    offset=base + 8).copy()
                out.setdefault(did, []).append(vec)
        return out, dim, total_vectors, total_docs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="new fp16 SoA mvbundle directory")
    ap.add_argument("--old_fivecs", required=True, help="legacy fp32 .fivecs file")
    ap.add_argument("--n_docs", type=int, default=100, help="how many docs to compare (from did=0)")
    ap.add_argument("--detailed_per_doc", action="store_true", help="print per-doc cosine summary")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    meta = json.load(open(bundle / "meta.json"))
    dim = meta["dim"]
    print(f"[bundle] {bundle.name}: dim={dim}, n_docs={meta['n_docs']}, n_tokens={meta['n_tokens']}, "
          f"doc_maxlen={meta.get('doc_maxlen')}")

    emb_new = np.memmap(bundle / "embeddings.fp16.bin", dtype=np.float16, mode="r").reshape(-1, dim)
    doclens_new = np.fromfile(bundle / "doclens.i32.bin", dtype=np.int32)
    docids_new = np.fromfile(bundle / "docids.i32.bin", dtype=np.int32)

    n = min(args.n_docs, len(doclens_new))
    cum = np.concatenate(([0], np.cumsum(doclens_new[:n], dtype=np.int64)))
    new_by_did = {int(docids_new[i]): np.asarray(emb_new[cum[i]:cum[i + 1]], dtype=np.float32)
                  for i in range(n)}

    print(f"[old] reading first {n} docs from {args.old_fivecs}")
    old_by_did, old_dim, old_total_vec, old_total_doc = read_old_fivecs_until_did(args.old_fivecs, n)
    print(f"[old] header: total_vectors={old_total_vec:,}, total_docs={old_total_doc:,}, dim={old_dim}")
    assert old_dim == dim, f"dim mismatch: new {dim} vs old {old_dim}"

    print()
    n_doclen_match = 0
    n_doclen_mismatch = 0
    cos_per_doc_summaries = []
    all_cos_chunks = []

    for did in range(n):
        new_vecs = new_by_did.get(did)
        old_vecs_list = old_by_did.get(did)
        if new_vecs is None or old_vecs_list is None:
            print(f"  doc {did}: missing — new={'present' if new_vecs is not None else 'NONE'} "
                  f"old={'present' if old_vecs_list is not None else 'NONE'}")
            n_doclen_mismatch += 1
            continue
        old_vecs = np.stack(old_vecs_list)
        if new_vecs.shape != old_vecs.shape:
            n_doclen_mismatch += 1
            if args.detailed_per_doc or n_doclen_mismatch <= 5:
                print(f"  doc {did}: SHAPE MISMATCH new {new_vecs.shape} vs old {old_vecs.shape}")
            continue
        n_doclen_match += 1
        cos = np.einsum("td,td->t", new_vecs, old_vecs)
        all_cos_chunks.append(cos)
        if args.detailed_per_doc:
            cos_per_doc_summaries.append((did, cos.size, float(cos.min()), float(cos.mean())))

    print()
    print(f"[result] doclen match: {n_doclen_match}/{n}   doclen mismatch: {n_doclen_mismatch}/{n}")

    if all_cos_chunks:
        all_cos = np.concatenate(all_cos_chunks)
        n_total = all_cos.size
        print(f"[result] tokens compared (matched docs only): {n_total:,}")
        print(f"  cosine    min   = {all_cos.min():.6f}")
        print(f"  cosine    mean  = {all_cos.mean():.6f}")
        print(f"  cosine    median= {np.median(all_cos):.6f}")
        print(f"  cosine    max   = {all_cos.max():.6f}")
        for thr in (0.9999, 0.999, 0.99, 0.9):
            n_below = int((all_cos < thr).sum())
            pct = n_below / n_total * 100
            print(f"  cosine < {thr}: {n_below:,} / {n_total:,}  ({pct:.4f}%)")
    else:
        print("[result] no tokens to compare (all docs mismatched)")

    if args.detailed_per_doc:
        print("\n[per-doc] (did, ntokens, cos_min, cos_mean)")
        for did, n_tok, cmin, cmean in cos_per_doc_summaries[:20]:
            print(f"  doc {did:>4}  n={n_tok:>4}  min={cmin:.6f}  mean={cmean:.6f}")


if __name__ == "__main__":
    main()
