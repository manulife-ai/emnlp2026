#!/usr/bin/env python3
"""Standalone reproduction of the conservative-anchor rows in Track B Table 3.

Paper: "When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval
for LLM Agents", Table 3 (conservative embedding-anchor regularization across
three Track-B data configurations).

This file is self-contained: it validates the release, fine-tunes conservative
LoRA with the embedding-anchor regularizer, and evaluates retrieval over the
full skill pool. It does not import anything from the ``scripts/synth`` pipeline.

Inputs
------
1. Track B data release (``data/trackBdata``):
       train.parquet, val.parquet,
       eval_set.parquet (Ring 1, SkillsBench real, n=21 queries),
       synthetic_eval_set.parquet (Ring 2, held-out-skill synthetic, n=2414),
       ood_eval_set.parquet (Ring 3, Terminal-Bench 2 OOD, n=10)
2. Skill catalog + retrieval index (not part of the data release):
       skills_meta.jsonl  and  index/skills.db

Data configurations (paper naming)
----------------------------------
  A  full mix
  B  skill-first synthetic rows removed
  C  only skill-first synthetic rows (10% of query_ids held out for validation)

Conservative LoRA recipe (Table 3): r=8, alpha=16, attention-only, lr=5e-6,
1 epoch, lambda=0.1.

Examples
--------
Validate the data release only::

    python scripts/release/trackb_cons_anchor_example.py --validate-only

Reproduce the conservative-anchor rows for all configurations::

    python scripts/release/trackb_cons_anchor_example.py \
        --data-dir data/trackBdata --out-dir ./output/trackb_table3 \
        --configs A,B,C

Every stage checkpoints to ``--out-dir``; re-running resumes instead of
recomputing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "trackBdata"
DEFAULT_OUT_DIR = REPO_ROOT / "data"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"
DEFAULT_SKILLS_META = REPO_ROOT / "data" / "skills_meta.jsonl"

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

SEED = 42
TOP_K = 10
TEMPERATURE = 0.05
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
EFFECTIVE_BATCH = 32
MICRO_BATCH = 8
MAX_LEN_QUERY = 512
MAX_LEN_SKILL = 384
MAX_SKILL_TEXT_CHARS = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Qwen3-Embedding official retrieval instruction format.
INSTRUCT_TASK = "Given a task description, retrieve relevant skills"

# Conservative recipe (Table 3).
CONS_LORA_R = 8
CONS_LORA_ALPHA = 16
CONS_LORA_DROPOUT = 0.1
CONS_LR = 5e-6
CONS_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Ring 1 / 2 / 3 use the paper's numbering. The released filenames follow the
# earlier internal "ring 1 / 1.5 / 2" naming.
RING_FILES = {
    "ring1_skillsbench_real": "eval_set.parquet",
    "ring2_synthetic_heldout": "synthetic_eval_set.parquet",
    "ring3_terminalbench_ood": "ood_eval_set.parquet",
}
RING_ORDER = list(RING_FILES)

EXPECTED_ROWS = {
    "positives.parquet": 255,
    "negatives.parquet": 1239,
    "task_positive_sets.parquet": 68,
    "train.parquet": 13271,
    "val.parquet": 556,
    "eval_set.parquet": 78,
    "synthetic_eval_set.parquet": 4023,
    "ood_eval_set.parquet": 15,
}
TRAIN_REQUIRED_COLUMNS = {
    "query_id", "positive_set_id", "query_text", "positive_skill_id",
    "positive_skill_text", "positive_weight", "negatives",
}
EVAL_REQUIRED_COLUMNS = {"query_id", "query_text", "positive_skill_id", "positive_set_id"}

PAPER_TABLE3 = {
    "A/anchor": (0.5400, 0.6083, 0.8500),
    "B/anchor": (0.5554, 0.5685, 0.8500),
    "C/anchor": (0.5463, 0.5626, 0.8500),
}
# Ring-1/3 are tiny (21 and 10 queries); one task moves Recall@10 by ~5%.
MATCH_TOL = {"ring1_skillsbench_real": 0.05,
             "ring2_synthetic_heldout": 0.02,
             "ring3_terminalbench_ood": 0.10}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data release: validation and loading
# ─────────────────────────────────────────────────────────────────────────────

def validate_release(data_dir: Path) -> dict[str, Any]:
    """Check that every released file exists with the locked row count/schema."""
    report: dict[str, Any] = {"root": str(data_dir), "files": {}, "valid": True}
    for name, expected in EXPECTED_ROWS.items():
        path = data_dir / name
        item: dict[str, Any] = {"exists": path.is_file(), "expected_rows": expected}
        if path.is_file():
            frame = pd.read_parquet(path)
            item["rows"] = len(frame)
            item["row_count_matches"] = len(frame) == expected
            if name in {"train.parquet", "val.parquet"}:
                required = TRAIN_REQUIRED_COLUMNS
            elif name.endswith("eval_set.parquet"):
                required = EVAL_REQUIRED_COLUMNS
            else:
                required = set()
            item["missing_columns"] = sorted(required - set(frame.columns))
            item["unique_queries"] = int(frame["query_id"].nunique()) if "query_id" in frame else None
        report["files"][name] = item
        report["valid"] &= bool(item["exists"]) and bool(item.get("row_count_matches"))
        report["valid"] &= not item.get("missing_columns")
    return report


def print_validation(report: dict[str, Any]) -> None:
    print(f"\n=== Track B release validation ({report['root']}) ===")
    print(f"{'file':<32}{'rows':>8}{'expected':>10}{'queries':>10}  status")
    for name, item in report["files"].items():
        ok = item["exists"] and item.get("row_count_matches") and not item.get("missing_columns")
        print(f"{name:<32}{item.get('rows', 0):>8}{item['expected_rows']:>10}"
              f"{item.get('unique_queries') or 0:>10}  {'OK' if ok else 'FAIL'}")
        if item.get("missing_columns"):
            print(f"    missing columns: {item['missing_columns']}")
    print(f"release_valid = {report['valid']}")


_TIER_COLS = ("positive_tier", "tier")
_SOURCE_COL = "query_source"
_SKILL_FIRST = "skill_first_synth"


def _tier_col(df: pd.DataFrame) -> str | None:
    return next((c for c in _TIER_COLS if c in df.columns), None)


def _to_triples(df: pd.DataFrame, tier_col: str | None) -> list[dict]:
    triples: list[dict] = []
    for _, r in df.iterrows():
        negs = r["negatives"]
        if isinstance(negs, np.ndarray):
            negs = negs.tolist()
        neg_texts = [str(n["skill_text"]) for n in (negs or [])]
        if not neg_texts:
            continue
        tier_val = r.get(tier_col) if tier_col else None
        triples.append({
            "query_id": str(r["query_id"]),
            "positive_set_id": str(r["positive_set_id"]),
            "query_text": str(r["query_text"]),
            "positive_text": str(r["positive_skill_text"]),
            "positive_weight": float(r["positive_weight"]),
            "negative_texts": neg_texts,
            "tier": str(tier_val) if tier_val is not None else "?",
        })
    return triples


def _holdout_split(df: pd.DataFrame, frac: float, seed: int, keep: str) -> pd.DataFrame:
    """Split by ``query_id`` so a query never straddles train and val."""
    rng = np.random.default_rng(seed)
    qids = df["query_id"].drop_duplicates().to_numpy()
    rng.shuffle(qids)
    n_hold = max(1, int(round(len(qids) * frac)))
    held = set(qids[:n_hold].tolist())
    mask = df["query_id"].isin(held)
    return df[mask].copy() if keep == "holdout" else df[~mask].copy()


def load_config_triples(data_dir: Path, config: str, split: str, seed: int = SEED) -> list[dict]:
    """Return contrastive triples for paper config A/B/C and split train/val.

    A  full mix; B  skill-first rows dropped; C  only skill-first rows, with 10%
    of the *training* query_ids carved out as the validation split (config C has
    no natural validation set in ``val.parquet``).
    """
    if config not in {"A", "B", "C"}:
        raise ValueError(f"config must be A, B or C (got {config!r})")
    if split not in {"train", "val"}:
        raise ValueError(f"split must be train or val (got {split!r})")

    if config == "C":
        df = pd.read_parquet(data_dir / "train.parquet")
        if _SOURCE_COL not in df.columns:
            raise ValueError(f"config C needs a {_SOURCE_COL!r} column in train.parquet")
        df = df[df[_SOURCE_COL] == _SKILL_FIRST].copy()
        df = _holdout_split(df, 0.10, seed, keep="holdout" if split == "val" else "train")
    else:
        df = pd.read_parquet(data_dir / f"{split}.parquet")
        if config == "B":
            if _SOURCE_COL not in df.columns:
                raise ValueError(f"config B needs a {_SOURCE_COL!r} column in {split}.parquet")
            df = df[df[_SOURCE_COL] != _SKILL_FIRST].copy()

    triples = _to_triples(df, _tier_col(df))
    print(f"[data] config {config} / {split}: {len(df)} rows → {len(triples)} triples "
          f"({df['query_id'].nunique()} queries)")
    return triples


# ─────────────────────────────────────────────────────────────────────────────
# 2. Skill catalog and corpus
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus_names(index_dir: Path) -> list[str]:
    with sqlite3.connect(index_dir / "skills.db") as conn:
        rows = conn.execute("SELECT rowid, name FROM skills ORDER BY rowid").fetchall()
    return [str(r[1]) for r in rows]


def load_skill_catalog(skills_meta: Path, skills_db: Path) -> dict[str, str]:
    """Map every skill alias to the text that gets encoded into the index."""
    snippets: dict[str, str] = {}
    if skills_db.exists():
        with sqlite3.connect(skills_db) as conn:
            for skill_id, name, snippet in conn.execute(
                    "SELECT skill_id, name, skill_md_snippet FROM skills"):
                if not snippet:
                    continue
                for key in (skill_id, name):
                    if key and str(key) not in snippets:
                        snippets[str(key)] = str(snippet)

    records: list[dict] = []
    with skills_meta.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("installs") or 0, reverse=True)

    catalog: dict[str, str] = {}
    for record in records:
        skill_name = (record.get("skill_name") or record.get("skill_id")
                      or record.get("skillId") or record.get("name"))
        if not skill_name:
            continue
        description = str(record.get("description") or "")
        snippet = ""
        for key in (record.get("skill_id"), record.get("skillId"),
                    record.get("name"), skill_name):
            if key and str(key) in snippets:
                snippet = snippets[str(key)]
                break
        lines = [f"Skill: {skill_name}"]
        if description:
            lines.append(f"Description: {description.strip()}")
        if snippet:
            lines.append(f"Snippet: {snippet.strip()}")
        text = "\n".join(lines)[:MAX_SKILL_TEXT_CHARS]
        for key in (record.get("skill_name"), record.get("skill_id"),
                    record.get("skillId"), record.get("name")):
            if key and str(key) not in catalog:
                catalog[str(key)] = text
    print(f"[catalog] {len(catalog)} skill aliases from {skills_meta.name}")
    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encoding
# ─────────────────────────────────────────────────────────────────────────────

def format_query(text: str) -> str:
    return f"Instruct: {INSTRUCT_TASK}\nQuery: {text}"


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-pad token regardless of which side the tokenizer pads."""
    if bool(mask[:, -1].all().item()):
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


def encode_batch(model, tok, texts: list[str], is_query: bool, max_len: int) -> torch.Tensor:
    if is_query:
        texts = [format_query(t) for t in texts]
    enc = tok(texts, padding=True, truncation=True,
              max_length=max_len, return_tensors="pt").to(DEVICE)
    out = model(**enc)
    emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
    return F.normalize(emb, p=2, dim=-1)


def load_tokenizer(model_name: str = BASE_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)


def encode_corpus(model, tok, names: list[str], catalog: dict[str, str],
                  batch_size: int = 32) -> np.ndarray:
    model.eval()
    embs: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            chunk = names[i:i + batch_size]
            texts = [(catalog.get(n) or f"Skill: {n}")[:MAX_SKILL_TEXT_CHARS] for n in chunk]
            embs.append(encode_batch(model, tok, texts, False, MAX_LEN_SKILL).float().cpu().numpy())
    print(f"  [encode] {len(names)} skills in {time.time() - t0:.0f}s")
    return np.vstack(embs)


def encode_queries(model, tok, texts: list[str], batch_size: int = 16) -> np.ndarray:
    model.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            embs.append(encode_batch(model, tok, texts[i:i + batch_size],
                                     True, MAX_LEN_QUERY).float().cpu().numpy())
    return np.vstack(embs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batching and losses
# ─────────────────────────────────────────────────────────────────────────────

def group_batches(triples: list[dict], micro_batch: int, seed: int) -> list[list[int]]:
    """Co-batch rows that share a positive_set_id so the loss can mask them."""
    rng = random.Random(seed)
    by_set: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(triples):
        by_set[t["positive_set_id"]].append(i)
    set_ids = list(by_set)
    rng.shuffle(set_ids)

    batches: list[list[int]] = []
    current: list[int] = []
    for sid in set_ids:
        group = by_set[sid][:]
        rng.shuffle(group)
        for i in range(0, len(group), micro_batch):
            chunk = group[i:i + micro_batch]
            if len(current) + len(chunk) > micro_batch:
                if current:
                    batches.append(current)
                current = chunk
            else:
                current.extend(chunk)
            if len(current) == micro_batch:
                batches.append(current)
                current = []
    if current:
        batches.append(current)
    return batches


def _candidate_scores(q_emb, pos_emb, neg_emb, n_neg: int,
                      positive_set_ids: list[str]) -> torch.Tensor:
    """(B, 1 + n_neg + B) scores: own positive, own hard negatives, in-batch positives."""
    B, d = q_emb.shape
    own_block = torch.cat([pos_emb.unsqueeze(1), neg_emb.view(B, n_neg, d)], dim=1)
    own = torch.einsum("bd,bnd->bn", q_emb, own_block) / TEMPERATURE
    cross = (q_emb @ pos_emb.t()) / TEMPERATURE
    sid = np.array(positive_set_ids)
    same = torch.tensor(sid[:, None] == sid[None, :], device=q_emb.device)
    cross = cross.masked_fill(same, float("-inf"))
    return torch.cat([own, cross], dim=1)


def mnrl_loss(q_emb, pos_emb, neg_emb, n_neg: int,
              positive_set_ids: list[str], pos_weights) -> torch.Tensor:
    scores = _candidate_scores(q_emb, pos_emb, neg_emb, n_neg, positive_set_ids)
    target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
    per_row = F.cross_entropy(scores, target, reduction="none")
    return (per_row * pos_weights).mean()


def anchor_loss(pos_emb: torch.Tensor, frozen_pos_emb: torch.Tensor) -> torch.Tensor:
    """Feature-space L2 anchor on in-batch positive skill embeddings (Eq. 1)."""
    return F.mse_loss(pos_emb, frozen_pos_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(lora_r: int | None = None, lora_alpha: int | None = None):
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    r = lora_r or CONS_LORA_R
    alpha = lora_alpha or CONS_LORA_ALPHA

    print(f"[model] {BASE_MODEL} conservative-anchor LoRA r={r} alpha={alpha} "
          f"dropout={CONS_LORA_DROPOUT} targets=attention-only")
    tok = load_tokenizer()
    base = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                     trust_remote_code=True).to(DEVICE)
    model = get_peft_model(base, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=CONS_LORA_DROPOUT, bias="none",
        target_modules=CONS_TARGETS, task_type="FEATURE_EXTRACTION",
    ))
    model.print_trainable_parameters()
    return model, tok


def trainable_named_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def load_adapter(ckpt_dir: Path):
    from transformers import AutoModel
    from peft import PeftModel
    base = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                     trust_remote_code=True).to(DEVICE)
    return PeftModel.from_pretrained(base, str(ckpt_dir)).to(DEVICE), load_tokenizer()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training
# ─────────────────────────────────────────────────────────────────────────────

def _batch_tensors(model, tok, rows: list[dict]):
    n_neg = max(len(r["negative_texts"]) for r in rows)
    neg_texts: list[str] = []
    for r in rows:
        negs = r["negative_texts"][:n_neg]
        if len(negs) < n_neg:
            negs = negs + [negs[-1]] * (n_neg - len(negs))
        neg_texts.extend(negs)
    q_emb = encode_batch(model, tok, [r["query_text"] for r in rows], True, MAX_LEN_QUERY)
    pos_emb = encode_batch(model, tok, [r["positive_text"] for r in rows], False, MAX_LEN_SKILL)
    neg_emb = encode_batch(model, tok, neg_texts, False, MAX_LEN_SKILL)
    pos_w = torch.tensor([r["positive_weight"] for r in rows],
                         dtype=q_emb.dtype, device=DEVICE)
    return q_emb, pos_emb, neg_emb, n_neg, pos_w, neg_texts


def val_loss(model, tok, val_triples: list[dict], n_batches: int = 20) -> float:
    if not val_triples:
        return float("nan")
    model.eval()
    rng = random.Random(SEED)
    sampled = rng.sample(val_triples, min(len(val_triples), n_batches * MICRO_BATCH))
    losses: list[float] = []
    with torch.no_grad():
        for idxs in group_batches(sampled, MICRO_BATCH, SEED)[:n_batches]:
            rows = [sampled[i] for i in idxs]
            if max(len(r["negative_texts"]) for r in rows) == 0:
                continue
            q, p, n, n_neg, w, _ = _batch_tensors(model, tok, rows)
            losses.append(float(mnrl_loss(q, p, n, n_neg, [r["positive_set_id"] for r in rows], w)))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_one(config: str, train_triples: list[dict], val_triples: list[dict],
              ckpt_dir: Path, lam: float, epochs: int, save_every: int,
              lora_r: int | None, lora_alpha: int | None, lr: float | None) -> list[dict]:
    """Fine-tune one configuration with conservative embedding-anchor LoRA."""
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "ckpt_log.json"
    if (ckpt_dir / "_complete.json").exists():
        print(f"[skip] {config}/{method}: completion marker found")
        return json.loads(log_path.read_text())

    model, tok = build_model(lora_r, lora_alpha)
    model.train()

    learning_rate = lr if lr is not None else CONS_LR
    batches = group_batches(train_triples, MICRO_BATCH, SEED)
    grad_accum = max(1, EFFECTIVE_BATCH // MICRO_BATCH)
    total_micro = len(batches) * epochs
    total_steps = math.ceil(total_micro / grad_accum)
    print(f"[train] {config}/anchor: {len(batches)} micro-batches x {epochs} epoch(s) "
          f"→ {total_steps} optim steps (lr={learning_rate}, lambda={lam})")

    optim = AdamW([param for _, param in trainable_named_params(model)],
                  lr=learning_rate, weight_decay=WEIGHT_DECAY)
    sched = get_cosine_schedule_with_warmup(
        optim, int(round(total_steps * WARMUP_RATIO)), total_steps)

    ckpt_log: list[dict] = []
    task_window: deque[float] = deque(maxlen=grad_accum)
    reg_window: deque[float] = deque(maxlen=grad_accum)
    step, mb_seen, t0 = 0, 0, time.time()

    for _ in range(epochs):
        for idxs in batches:
            rows = [train_triples[i] for i in idxs]
            if max(len(r["negative_texts"]) for r in rows) == 0:
                continue
            mb_seen += 1
            psids = [r["positive_set_id"] for r in rows]
            q, p, n, n_neg, w, neg_texts = _batch_tensors(model, tok, rows)
            loss_task = mnrl_loss(q, p, n, n_neg, psids, w)

            with model.disable_adapter(), torch.no_grad():
                frozen_pos = encode_batch(model, tok, [r["positive_text"] for r in rows],
                                          False, MAX_LEN_SKILL).detach()
            loss_reg = anchor_loss(p, frozen_pos.to(p.dtype))

            loss = loss_task + lam * loss_reg
            (loss / grad_accum).backward()
            task_window.append(float(loss_task))
            reg_window.append(float(loss_reg))

            if mb_seen % grad_accum == 0 or mb_seen == total_micro:
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == 1:
                    print(f"  step {step}/{total_steps} task={np.mean(task_window):.4f} "
                          f"reg={np.mean(reg_window):.4f} ({time.time() - t0:.0f}s)")
                if step % save_every == 0 or step == total_steps:
                    sub = ckpt_dir / f"step_{step}"
                    sub.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(sub))
                    vl = val_loss(model, tok, val_triples)
                    ckpt_log.append({"step": step, "val_loss": float(vl),
                                     "train_loss": float(np.mean(task_window)),
                                     "reg_loss": float(np.mean(reg_window)),
                                     "path": str(sub)})
                    log_path.write_text(json.dumps(ckpt_log, indent=2))
                    print(f"  [ckpt] step {step} val_loss={vl:.4f} → {sub}")

    tok.save_pretrained(str(ckpt_dir))
    log_path.write_text(json.dumps(ckpt_log, indent=2))
    (ckpt_dir / "_complete.json").write_text(json.dumps(
        {"method": "anchor", "config": config, "total_steps": total_steps,
         "lambda": lam, "lr": learning_rate, "finished_at": time.time()}, indent=2))
    del model, tok
    torch.cuda.empty_cache()
    return ckpt_log


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _read_ring(path: Path) -> tuple[list[str], dict[str, str], dict[str, set[str]]]:
    df = pd.read_parquet(path)
    if "query_text" not in df.columns and "task_text" in df.columns:
        df = df.rename(columns={"task_text": "query_text"})
    qtext: dict[str, str] = {}
    gold: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        qid = str(r.get("query_id") or r.get("task_name"))
        qtext[qid] = str(r["query_text"])
        if "positive_skill_id" in df.columns:
            gold.setdefault(qid, set()).add(str(r["positive_skill_id"]))
    return list(qtext), qtext, gold


def _metrics(ranked: list[str], g: set[str], k: int) -> dict:
    topk = set(ranked[:k])
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(g), k)))
    dcg = sum(1.0 / math.log2(r + 2) for r, n in enumerate(ranked[:k]) if n in g)
    mrr = next((1.0 / (r + 1) for r, n in enumerate(ranked) if n in g), 0.0)
    return {"cardinality": len(g),
            "set_recall_at_k": len(topk & g) / len(g),
            "soft_ndcg_at_k": dcg / ideal if ideal > 0 else 0.0,
            "mrr": mrr,
            "coverage_at_k": len(topk & g)}


def eval_ring(corpus_emb: np.ndarray, names: list[str], model, tok,
              ring_path: Path, k: int = TOP_K, max_queries: int | None = None) -> list[dict]:
    qids, qtext, gold = _read_ring(ring_path)
    if max_queries:
        qids = qids[:max_queries]
    q_emb = encode_queries(model, tok, [qtext[q] for q in qids])
    top_idx = np.argsort(-(q_emb @ corpus_emb.T), axis=1)[:, :max(k, 20)]
    per_q: list[dict] = []
    for i, qid in enumerate(qids):
        g = gold.get(qid, set())
        if not g:
            continue
        per_q.append({"query_id": qid, **_metrics([names[j] for j in top_idx[i]], g, k)})
    return per_q


def select_checkpoint(ckpt_log: list[dict], ring2_path: Path, names: list[str],
                      catalog: dict[str, str], select_queries: int = 500) -> dict:
    """Top-3 by validation loss, then pick the best Ring-2 Recall@10."""
    usable = [c for c in ckpt_log
              if isinstance(c.get("val_loss"), (int, float)) and not math.isnan(c["val_loss"])]
    if not usable:
        return {"step": None, "path": None, "selection_mode": "no_checkpoints"}
    top3 = sorted(usable, key=lambda c: c["val_loss"])[:3]
    print(f"[select] top-3 by val_loss: {[(c['step'], round(c['val_loss'], 4)) for c in top3]}")
    if len(top3) == 1 or not ring2_path.exists():
        w = top3[0]
        return {"step": w["step"], "path": w["path"], "val_loss": w["val_loss"],
                "selection_mode": "val_loss_only"}

    best: dict | None = None
    for c in top3:
        model, tok = load_adapter(Path(c["path"]))
        corpus = encode_corpus(model, tok, names, catalog)
        per_q = eval_ring(corpus, names, model, tok, ring2_path, max_queries=select_queries)
        score = float(np.mean([x["set_recall_at_k"] for x in per_q])) if per_q else float("nan")
        print(f"  step {c['step']}: Ring-2 Recall@{TOP_K} = {score:.4f}")
        if not math.isnan(score) and (best is None or score > best["ring2_score"]):
            best = {"step": c["step"], "path": c["path"], "val_loss": c["val_loss"],
                    "ring2_score": score, "selection_mode": "ring2_recall"}
        del model, tok, corpus
        torch.cuda.empty_cache()
    if best is None:
        w = top3[0]
        return {"step": w["step"], "path": w["path"], "val_loss": w["val_loss"],
                "selection_mode": "ring2_nan_fallback"}
    print(f"[select] winner: step {best['step']}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 8. Result store and reporting
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict[str, dict[str, list[dict]]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_results(path: Path, results: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(path)


def ring_mean(results: dict, variant: str, ring: str, metric: str = "set_recall_at_k") -> float:
    rows = results.get(variant, {}).get(ring, [])
    return float(np.mean([r[metric] for r in rows])) if rows else float("nan")


def _fmt(x: float) -> str:
    return "  n/a " if math.isnan(x) else f"{x:.4f}"


def print_tables(results: dict, metric: str = "set_recall_at_k") -> dict:
    hdr = f"{'variant':<26}{'Ring 1':>10}{'Ring 2':>10}{'Ring 3':>10}"
    print("\n" + "=" * 78)
    print(f"Recall@{TOP_K}   Ring 1 = SkillsBench (real, n=21)   "
          f"Ring 2 = synthetic held-out (n=2414)   Ring 3 = Terminal-Bench 2 (real, n=10)")
    print("=" * 78)

    print("\n--- Table 3: conservative embedding-anchor LoRA ---")
    print(hdr + f"{'dR1':>9}{'dR2':>9}{'dR3':>9}  match")
    comparison: dict[str, Any] = {}
    for variant, paper in PAPER_TABLE3.items():
        vals = [ring_mean(results, variant, r, metric) for r in RING_ORDER]
        if all(math.isnan(v) for v in vals):
            continue
        deltas = [v - p for v, p in zip(vals, paper)]
        ok = all((not math.isnan(d)) and abs(d) <= MATCH_TOL[r]
                 for d, r in zip(deltas, RING_ORDER))
        comparison[variant] = {
            "reproduced": dict(zip(RING_ORDER, vals)),
            "paper": dict(zip(RING_ORDER, paper)),
            "delta": dict(zip(RING_ORDER, deltas)),
            "within_tolerance": bool(ok),
        }
        print(f"{variant:<26}" + "".join(_fmt(v).rjust(10) for v in vals)
              + "".join(("   n/a  " if math.isnan(d) else f"{d:+.4f}").rjust(9) for d in deltas)
              + ("   YES" if ok else "   NO"))

    if comparison:
        n_ok = sum(1 for c in comparison.values() if c["within_tolerance"])
        print(f"\nTable 3 rows within tolerance "
              f"(R1 +/-{MATCH_TOL['ring1_skillsbench_real']}, "
              f"R2 +/-{MATCH_TOL['ring2_synthetic_heldout']}, "
              f"R3 +/-{MATCH_TOL['ring3_terminalbench_ood']}): {n_ok}/{len(comparison)}")
    else:
        print("\n(no conservative-anchor variants evaluated yet — run with --configs A,B,C)")
    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# 9. Driver
# ─────────────────────────────────────────────────────────────────────────────

def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--skills-meta", type=Path, default=DEFAULT_SKILLS_META)
    ap.add_argument("--configs", default="A,B,C", help="subset of A,B,C (paper naming)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.1,
                    help="regularizer weight (paper uses 0.1)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=100, help="optim steps between checkpoints")
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-eval-queries", type=int, default=None,
                    help="cap queries per ring (smoke tests only; changes the numbers)")
    ap.add_argument("--validate-only", action="store_true",
                    help="check the data release and exit")
    ap.add_argument("--report-only", action="store_true",
                    help="reprint tables from an existing results.json and exit")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.json"

    report = validate_release(args.data_dir)
    print_validation(report)
    if args.validate_only:
        return 0 if report["valid"] else 1
    if not report["valid"]:
        print("\n[error] data release did not validate; refusing to run experiments.")
        return 1

    if args.report_only:
        results = load_results(results_path)
        comparison = print_tables(results)
        (args.out_dir / "table3_comparison.json").write_text(json.dumps(comparison, indent=2))
        return 0

    configs = _csv(args.configs)
    bad = set(configs) - {"A", "B", "C"}
    if bad:
        raise SystemExit(f"--configs must be a subset of A,B,C (got {sorted(bad)})")

    db_path = args.index_dir / "skills.db"
    for required in (db_path, args.skills_meta):
        if not required.exists():
            raise SystemExit(f"[error] missing required file: {required}\n"
                             f"Download the skill catalog and index first "
                             f"(skill_router/scripts/download_search_index.py).")

    rings = {name: args.data_dir / fname for name, fname in RING_FILES.items()}
    for name, path in rings.items():
        if not path.exists():
            raise SystemExit(f"[error] missing ring file for {name}: {path}")

        print(f"\n[env] device={DEVICE} torch={torch.__version__} "
            f"configs={configs} method=anchor")
    names = load_corpus_names(args.index_dir)
    catalog = load_skill_catalog(args.skills_meta, db_path)
    print(f"[corpus] {len(names)} indexed skills")

    results = load_results(results_path)

    def record(variant: str, ring: str, per_q: list[dict]) -> None:
        results.setdefault(variant, {})[ring] = per_q
        save_results(results_path, results)
        mean = float(np.mean([r["set_recall_at_k"] for r in per_q])) if per_q else float("nan")
        print(f"  {variant} / {ring}: n={len(per_q)} Recall@{TOP_K}={_fmt(mean).strip()}")

    def eval_model(variant: str, model, tok) -> None:
        pending = [r for r in RING_ORDER if r not in results.get(variant, {})]
        if not pending:
            print(f"[skip] {variant}: all rings already evaluated")
            return
        corpus = encode_corpus(model, tok, names, catalog)
        for ring in pending:
            record(variant, ring, eval_ring(corpus, names, model, tok, rings[ring],
                                            max_queries=args.max_eval_queries))
        del corpus
        torch.cuda.empty_cache()

    # --- Conservative embedding-anchor fine-tuning --------------------------
    selections: dict[str, dict] = {}
    for config in configs:
        variant = f"{config}/anchor"
        if all(r in results.get(variant, {}) for r in RING_ORDER):
            print(f"\n[skip] {variant}: already evaluated")
            continue
        print(f"\n{'=' * 78}\n=== TRAIN {variant} ===\n{'=' * 78}")
        train_triples = load_config_triples(args.data_dir, config, "train", args.seed)
        val_triples = load_config_triples(args.data_dir, config, "val", args.seed)
        ckpt_dir = args.out_dir / "adapters" / config / "anchor"
        ckpt_log = train_one(config, train_triples, val_triples, ckpt_dir,
                             lam=args.lam, epochs=args.epochs, save_every=args.save_every,
                             lora_r=args.lora_r, lora_alpha=args.lora_alpha, lr=args.lr)
        del train_triples, val_triples
        torch.cuda.empty_cache()

        sel_path = ckpt_dir / "selected.json"
        if sel_path.exists():
            selection = json.loads(sel_path.read_text())
        else:
            selection = select_checkpoint(ckpt_log, rings["ring2_synthetic_heldout"],
                                          names, catalog)
            sel_path.write_text(json.dumps(selection, indent=2))
        selections[variant] = selection
        if not selection.get("path"):
            print(f"[warn] {variant}: no usable checkpoint; skipping evaluation")
            continue

        print(f"\n=== EVAL {variant} (step {selection['step']}) ===")
        model, tok = load_adapter(Path(selection["path"]))
        eval_model(variant, model, tok)
        del model, tok
        torch.cuda.empty_cache()

    if selections:
        (args.out_dir / "selections.json").write_text(json.dumps(selections, indent=2))

    comparison = print_tables(results)
    (args.out_dir / "table3_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"\n[ok] per-query results → {results_path}")
    print(f"[ok] Table 3 comparison → {args.out_dir / 'table3_comparison.json'}")
    return 0


if __name__ == "__main__":
    main()
