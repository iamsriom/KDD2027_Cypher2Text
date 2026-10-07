#!/usr/bin/env python3
"""Stage A: Deterministic-vocab CSEncoder pretrain (variable-rename InfoNCE).

Builds a lex-sorted vocabulary with <UNK>=0, computes IDF, trains with in-batch
InfoNCE on (cypher, variable-renamed) pairs, and writes:

  data/models/token_to_id.json
  data/models/idf_scores_text2cypher.json
  data/models/csencoder_pretrain.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csencoder.encoding import (  # noqa: E402
    MAX_LEN,
    attach_idf,
    build_encoder,
    encode_views,
    prepare_views,
    save_vocab,
)
from csencoder.preprocessing.graph_builder import CypherTokenizer  # noqa: E402
from csencoder.preprocessing.rarity_calc import IDFCalculator  # noqa: E402

try:
    from data.setup.config import DEFAULT_HF_DATASET_NAME, DEFAULT_HF_TOKEN_PATH, MODELS_DIR
except ImportError:
    DEFAULT_HF_DATASET_NAME = "neo4j/text2cypher-2024v1"
    DEFAULT_HF_TOKEN_PATH = Path.home() / "Huggingface_api.txt"
    MODELS_DIR = REPO_ROOT / "data" / "models"

LOGGER = logging.getLogger("train_csencoder_pretrain")

HIDDEN_DIM = 768
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.07
GRAD_CLIP = 1.0


class VariableRenamingAugmenter:
    def __init__(self) -> None:
        self.var_pattern = re.compile(r"\b([a-z][a-zA-Z0-9]*)\b")

    def augment(self, cypher: str) -> str:
        variables = set(self.var_pattern.findall(cypher.lower()))
        variables = {v for v in variables if len(v) <= 3 and v.isalpha()}
        if not variables:
            return cypher
        replacement_vars = list("xyzabcdefghijklmnopqrstuvw")
        available = [v for v in replacement_vars if v not in variables]
        augmented = cypher
        for old_var in sorted(variables):
            if not available:
                break
            new_var = available.pop(0)
            augmented = re.sub(
                r"\b" + re.escape(old_var) + r"\b",
                new_var,
                augmented,
                flags=re.IGNORECASE,
            )
        return augmented


class PairDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def ensure_hf_login(token_path: Path) -> None:
    token = os.getenv("HF_TOKEN")
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
    if os.getenv("HF_TOKEN"):
        from huggingface_hub import login as hf_login

        hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)


def load_train_cyphers(dataset_name: str, limit: int | None) -> List[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    col = "cypher" if "cypher" in ds.column_names else "query"
    cyphers = [c.strip() for c in ds[col] if isinstance(c, str) and c.strip()]
    if limit is not None:
        cyphers = cyphers[:limit]
    return cyphers


def build_vocabulary(queries: Sequence[str], tokenizer: CypherTokenizer) -> Dict[str, int]:
    tokens = set()
    for query in tqdm(queries, desc="Tokenizing train corpus"):
        tokens.update(tokenizer.tokenize(query))
    ordered = sorted(token for token in tokens if token != "<UNK>")
    token_to_id = {"<UNK>": 0}
    for index, token in enumerate(ordered, start=1):
        token_to_id[token] = index
    return token_to_id


def infonce_loss(embeddings: torch.Tensor, augmented: torch.Tensor, temperature: float) -> torch.Tensor:
    embeddings = F.normalize(embeddings, p=2, dim=1)
    augmented = F.normalize(augmented, p=2, dim=1)
    logits = embeddings @ augmented.t() / temperature
    labels = torch.arange(embeddings.size(0), device=embeddings.device)
    return F.cross_entropy(logits, labels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A: deterministic-vocab CSEncoder pretrain")
    parser.add_argument("--dataset-name", default=DEFAULT_HF_DATASET_NAME)
    parser.add_argument("--hf-token-path", type=Path, default=DEFAULT_HF_TOKEN_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Graph prep workers (0=serial)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    set_seed(args.seed)
    ensure_hf_login(args.hf_token_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    cyphers = load_train_cyphers(args.dataset_name, args.limit)
    LOGGER.info("Loaded %d train Cypher queries", len(cyphers))

    tokenizer = CypherTokenizer()
    token_to_id = build_vocabulary(cyphers, tokenizer)
    vocab_path = models_dir / "token_to_id.json"
    save_vocab(vocab_path, token_to_id, n_docs=len(cyphers), dataset_name=args.dataset_name)
    # Also write flat token map for older readers
    (models_dir / "vocab.json").write_text(
        json.dumps(
            {
                "unk_token": "<UNK>",
                "unk_id": 0,
                "vocab_size": len(token_to_id),
                "token_to_id": token_to_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Vocab size %d -> %s", len(token_to_id), vocab_path)

    idf_calc = IDFCalculator()
    idf_scores = idf_calc.compute_idf(cyphers)
    idf_path = models_dir / "idf_scores_text2cypher.json"
    idf_calc.save(idf_path)
    (models_dir / "idf_scores.json").write_text(
        json.dumps(idf_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("IDF tokens %d -> %s", len(idf_scores), idf_path)

    augmenter = VariableRenamingAugmenter()
    LOGGER.info("Building graph views (workers=%s)", args.workers)
    views = prepare_views(cyphers, workers=args.workers)
    aug_views = prepare_views([augmenter.augment(c) for c in cyphers], workers=args.workers)
    rows = [
        {"view": attach_idf(v, idf_scores), "aug_view": attach_idf(a, idf_scores)}
        for v, a in zip(views, aug_views)
    ]

    dataset = PairDataset(rows)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, max(1, len(dataset))),
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: batch,
        drop_last=len(dataset) > args.batch_size,
        num_workers=0,
    )

    model = build_encoder(len(token_to_id), dropout=DROPOUT, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    history: List[float] = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        started = time.perf_counter()
        for batch in tqdm(loader, desc=f"pretrain epoch {epoch}/{args.epochs}"):
            emb = encode_views(model, [b["view"] for b in batch], token_to_id, device)
            aug = encode_views(model, [b["aug_view"] for b in batch], token_to_id, device)
            loss = infonce_loss(emb, aug, TEMPERATURE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        avg = total / steps if steps else 0.0
        history.append(avg)
        LOGGER.info("Epoch %s/%s loss %.6f time %.1fs", epoch, args.epochs, avg, time.perf_counter() - started)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, models_dir / f"csencoder_pretrain_epoch_{epoch}.pt")
        # Prefer epoch 2 when available (selected pretrain checkpoint), else best loss
        if epoch == 2 or (best_state is None and epoch == 1):
            best_state = state
            best_loss = avg
        elif avg < best_loss:
            best_loss = avg
            if args.epochs < 2:
                best_state = state

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_path = models_dir / "csencoder_pretrain.pt"
    torch.save(best_state, out_path)
    (models_dir / "pretrain_history.json").write_text(
        json.dumps(
            {
                "loss_history": history,
                "selected": "epoch_2_if_available_else_best",
                "best_loss": best_loss,
                "vocab_size": len(token_to_id),
                "max_seq_len": MAX_LEN,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote pretrained checkpoint %s", out_path)


if __name__ == "__main__":
    main()
