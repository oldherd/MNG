#!/usr/bin/env python3
"""
encode.py — stream-encode a TSV (id\\ttext) into a multi-vector bundle.

Bundle layout (one directory per dataset/split):
  embeddings.fp16.bin   doc mode: contiguous fp16 token vectors  (N_tokens, dim)
  embeddings.fp32.bin   query mode: contiguous fp32 token vectors (N_tokens, dim)
  doclens.i32.bin       int32 per-row token count
  docids.i32.bin        int32 original row ids (doc id / query id)
  meta.json             magic / version / dtype / dim / counts / hashes / args

doc end follows ColBERTv2 native: encoder forward output is fp16 (model.doc()
ends with `.half()` on GPU). Query end keeps fp32 (model.query() has no .half();
its scoring is small + latency-sensitive + needs full precision for [MASK]
expansion vectors). Either file is mmap-ready: np.memmap(path, dtype=...).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

VANILLA_DEFAULT = ""
MAGIC = "mvbin"
VERSION = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["doc", "query"], required=True)
    p.add_argument("--input", required=True, help="TSV file: <id>\\t<text> per line")
    p.add_argument("--output", required=True, help="output bundle directory (created if missing)")
    p.add_argument("--ckpt", required=True, help="ColBERTv2 checkpoint directory")
    p.add_argument("--vanilla", default=VANILLA_DEFAULT, help="path to vanilla ColBERT repo (added to sys.path)")
    p.add_argument("--dim", type=int, default=128, help="ColBERT projection dim")
    p.add_argument("--doc_maxlen", type=int, default=220)
    p.add_argument("--query_maxlen", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--keep_padding", action="store_true",
                   help="(query mode) write padding vectors too — each query will be exactly query_maxlen vectors")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 → only encode first N rows (sanity check)")
    p.add_argument("--dataset", default="", help="metadata only: dataset name")
    p.add_argument("--split", default="", help="metadata only: split name (dev/test/etc)")
    return p.parse_args()


def sha256_of(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()

    if args.vanilla and args.vanilla not in sys.path:
        sys.path.insert(0, args.vanilla)
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] checkpoint={args.ckpt}  vanilla={args.vanilla}", flush=True)
    config = ColBERTConfig(
        checkpoint=args.ckpt,
        doc_maxlen=args.doc_maxlen,
        query_maxlen=args.query_maxlen,
    )
    model = Checkpoint(config.checkpoint, colbert_config=config).cuda()
    model.eval()
    if hasattr(model, "doc_tokenizer"):
        model.doc_tokenizer.doc_maxlen = args.doc_maxlen
    if hasattr(model, "query_tokenizer"):
        model.query_tokenizer.query_maxlen = args.query_maxlen

    is_query = (args.mode == "query")
    tokenizer = model.query_tokenizer if is_query else model.doc_tokenizer

    with open(args.input, "r", encoding="utf-8") as f:
        rows = []
        for ln in f:
            ln = ln.rstrip("\n").rstrip("\r")
            if not ln:
                continue
            rows.append(ln)
            if args.limit > 0 and len(rows) >= args.limit:
                break
    n_docs = len(rows)
    print(f"[input] {n_docs:,} rows ← {args.input}", flush=True)
    if n_docs == 0:
        raise SystemExit("empty input")

    target_dtype = torch.float32 if is_query else torch.float16
    np_dtype = np.float32 if is_query else np.float16
    bytes_per_val = 4 if is_query else 2
    emb_fname = "embeddings.fp32.bin" if is_query else "embeddings.fp16.bin"

    emb_path = out_dir / emb_fname
    docl_path = out_dir / "doclens.i32.bin"
    docids_path = out_dir / "docids.i32.bin"
    meta_path = out_dir / "meta.json"

    doclens = np.empty(n_docs, dtype=np.int32)
    docids = np.empty(n_docs, dtype=np.int32)
    n_tokens_total = 0
    t0 = time.time()
    last_log = t0

    with open(emb_path, "wb") as f_emb:
        for start in range(0, n_docs, args.batch_size):
            batch = rows[start : start + args.batch_size]
            ids, texts = [], []
            for j, ln in enumerate(batch):
                parts = ln.split("\t", 1)
                if len(parts) == 2:
                    try:
                        ids.append(int(parts[0]))
                    except ValueError:
                        ids.append(start + j)
                    texts.append(parts[1])
                else:
                    ids.append(start + j)
                    texts.append(parts[0])

            with torch.no_grad():
                input_ids, attn = tokenizer.tensorize(texts)
                if is_query:
                    embs = model.query(input_ids, attn)
                    if args.keep_padding:
                        mask = torch.ones_like(embs[..., 0]).bool()
                    else:
                        mask = attn.bool().to(embs.device)
                else:
                    embs, mask = model.doc(input_ids, attn, keep_dims="return_mask")

            embs_cpu = embs.detach().to(target_dtype).cpu().numpy()
            mask_cpu = mask.detach().cpu().numpy().astype(bool)
            assert embs_cpu.dtype == np_dtype, (embs_cpu.dtype, np_dtype)

            for j in range(len(ids)):
                valid_idx = np.where(mask_cpu[j])[0]
                ntok = int(valid_idx.size)
                vecs = embs_cpu[j, valid_idx, :]
                vecs.tofile(f_emb)
                row = start + j
                doclens[row] = ntok
                docids[row] = ids[j]
                n_tokens_total += ntok

            now = time.time()
            done = start + len(batch)
            if now - last_log >= 5.0 or done == n_docs:
                rate = done / max(now - t0, 1e-9)
                eta = (n_docs - done) / max(rate, 1e-9)
                print(f"[progress] {done:,}/{n_docs:,}  {rate:.1f} doc/s  ETA {eta:.0f}s  tokens={n_tokens_total:,}",
                      flush=True)
                last_log = now

    doclens.tofile(docl_path)
    docids.tofile(docids_path)

    meta = {
        "magic": MAGIC,
        "version": VERSION,
        "dtype": "float32" if is_query else "float16",
        "dim": args.dim,
        "n_tokens": int(n_tokens_total),
        "n_docs": int(n_docs),
        "mode": args.mode,
        "is_query": is_query,
        "keep_padding": bool(args.keep_padding) if is_query else False,
        "doc_maxlen": args.doc_maxlen,
        "query_maxlen": args.query_maxlen,
        "encoder_ckpt": str(args.ckpt),
        "input_tsv": str(args.input),
        "dataset": args.dataset,
        "split": args.split,
        "elapsed_sec": round(time.time() - t0, 2),
        "files": {
            emb_fname: {"size": emb_path.stat().st_size, "sha256": sha256_of(emb_path)},
            "doclens.i32.bin": {"size": docl_path.stat().st_size, "sha256": sha256_of(docl_path)},
            "docids.i32.bin": {"size": docids_path.stat().st_size, "sha256": sha256_of(docids_path)},
        },
    }

    expected_emb_size = n_tokens_total * args.dim * bytes_per_val
    if meta["files"][emb_fname]["size"] != expected_emb_size:
        raise RuntimeError(
            f"size mismatch: {emb_fname} is "
            f"{meta['files'][emb_fname]['size']} bytes, "
            f"expected {expected_emb_size}"
        )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] {n_tokens_total:,} tokens, {n_docs:,} docs, {meta['elapsed_sec']}s", flush=True)
    print(f"[output] {out_dir}/")


if __name__ == "__main__":
    main()
