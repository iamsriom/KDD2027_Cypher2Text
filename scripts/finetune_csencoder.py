#!/usr/bin/env python3
"""Stage B: weighted InfoNCE finetune on mined hard positives / hard negatives.

Default Neo4j recipe:
τ=0.07, batch 16, 4 epochs, main lr 1e-5, structural lr 3e-5, select epoch 1.
Writes data/models/csencoder_model.pt (selected epoch) plus per-epoch checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
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
    load_vocab,
    prepare_views,
    validate_view_cache_version,
)
from csencoder.preprocessing.graph_builder import FEATURE_CACHE_VERSION  # noqa: E402

try:
    from data.setup.config import MODELS_DIR
except ImportError:
    MODELS_DIR = REPO_ROOT / "data" / "models"

LOGGER = logging.getLogger("finetune_csencoder")

HIDDEN_DIM = 768
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
EPOCHS = 4
TAU = 0.07
RANK_MARGIN = 0.0
RANK_LAMBDA = 0.0
POS_WEIGHTS = {"p1": 0.35, "p2": 1.0, "p3": 0.35, "p4": 0.75, "p5": 0.75}
NEG_WEIGHTS = {"n1": 0.75, "n2": 0.75, "n3": 1.5, "n4": 1.5, "nc": 1.5}
STRUCTURAL_SLOTS = {"n3", "n4"}
SCHEMA_FIELDS = ("label", "relationship", "property", "schema")
GRAD_CLIP = 1.0
MAIN_LR = 1e-5
STRUCT_LR = 3e-5
SELECTED_EPOCH = 1


def _filled(item: dict | None) -> bool:
    return bool(item and item.get("cypher"))


def _slot(record: dict, name: str) -> dict | None:
    pool = record["positives"] if name.startswith("p") else record["hard_negatives"]
    for item in pool:
        if item.get("slot") == name:
            return item
    return None


def load_supervision(path: Path, expected: int | None = None) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"Expected {expected} supervision rows, found {len(rows)}")
    return rows


def build_examples(records: List[dict], strict_schema_positives: bool = False) -> List[dict]:
    examples = []
    skipped_no_negative = 0
    skipped_no_positive = 0
    for record in records:
        positives = []
        negatives = []
        seen_pos = set()
        seen_neg = set()

        def add_positive(item: dict | None, weight: float, kind: str) -> None:
            if not _filled(item):
                return
            if kind == "p1" and not item.get("changed", True):
                return
            if kind == "p3":
                scores = item.get("scores") or {}
                if float(scores.get("schema") or 0.0) < 0.80 or float(scores.get("operators") or 0.0) < 0.70:
                    return
            if strict_schema_positives and kind in {"p2", "p3", "p4", "p5"}:
                scores = item.get("scores") or {}
                if any(float(scores.get(name) or 0.0) < 0.80 for name in SCHEMA_FIELDS):
                    return
            key = ("p1", record["anchor_id"]) if kind == "p1" else ("train", int(item["id"]))
            if key in seen_pos:
                return
            seen_pos.add(key)
            positives.append({"key": key, "weight": weight})

        def add_negative(item: dict | None, weight: float) -> None:
            if not _filled(item):
                return
            if item.get("counterfactual"):
                key = ("cf", item["cypher"])
            elif item.get("id") is None:
                return
            else:
                key = ("train", int(item["id"]))
            if key in seen_pos:
                return
            if key in seen_neg:
                for existing in negatives:
                    if existing["key"] == key:
                        existing["weight"] = max(existing["weight"], weight)
                return
            seen_neg.add(key)
            negatives.append({"key": key, "weight": weight})

        add_positive(_slot(record, "p1"), POS_WEIGHTS["p1"], "p1")
        add_positive(_slot(record, "p2"), POS_WEIGHTS["p2"], "p2")
        add_positive(_slot(record, "p3"), POS_WEIGHTS["p3"], "p3")
        add_positive(_slot(record, "p4"), POS_WEIGHTS["p4"], "p4")
        add_positive(_slot(record, "p5"), POS_WEIGHTS["p5"], "p5")
        add_negative(_slot(record, "nc"), NEG_WEIGHTS["nc"])
        add_negative(_slot(record, "n1"), NEG_WEIGHTS["n1"])
        add_negative(_slot(record, "n2"), NEG_WEIGHTS["n2"])
        add_negative(_slot(record, "n3"), NEG_WEIGHTS["n3"])
        add_negative(_slot(record, "n4"), NEG_WEIGHTS["n4"])
        if not negatives:
            skipped_no_negative += 1
            continue
        if not positives:
            skipped_no_positive += 1
            continue
        has_structural = any(
            item.get("cypher") and item.get("slot") in STRUCTURAL_SLOTS for item in record["hard_negatives"]
        )
        examples.append(
            {
                "anchor": int(record["anchor_id"]),
                "positives": positives,
                "negatives": negatives,
                "structural": has_structural,
            }
        )
    LOGGER.info(
        "Usable anchors %s skipped_no_negative %s skipped_no_positive %s structural %s",
        len(examples),
        skipped_no_negative,
        skipped_no_positive,
        sum(1 for item in examples if item["structural"]),
    )
    return examples


def training_order(examples: List[dict]) -> List[int]:
    order = []
    for index, example in enumerate(examples):
        order.append(index)
        if example["structural"]:
            order.append(index)
    return order


class ExampleDataset(Dataset):
    def __init__(self, examples: List[dict], order: List[int]):
        self.examples = examples
        self.order = order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index: int) -> dict:
        return self.examples[self.order[index]]


def weighted_infonce(
    anchor: torch.Tensor,
    positives: torch.Tensor,
    pos_weight: torch.Tensor,
    negatives: torch.Tensor,
    neg_weight: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_logit = positives @ anchor / tau
    neg_logit = negatives @ anchor / tau
    neg_logged = neg_logit + torch.log(neg_weight)
    max_logit = torch.maximum(pos_logit.max(), neg_logged.max())
    neg_sum = torch.exp(neg_logged - max_logit).sum()
    pos_exp = torch.exp(pos_logit - max_logit)
    denom = torch.log(pos_exp + neg_sum) + max_logit
    losses = denom - pos_logit
    loss = (pos_weight * losses).sum() / pos_weight.sum()
    return loss, (positives @ anchor).detach(), (negatives @ anchor).detach()


def resolve(
    key: tuple,
    anchor: int,
    train_views: List[dict],
    p1_views: List[dict],
    cf_views: Dict[str, dict] | None = None,
) -> dict:
    kind, index = key
    if kind == "p1":
        return p1_views[anchor]
    if kind == "cf":
        if cf_views is None or index not in cf_views:
            raise RuntimeError("Missing counterfactual view")
        return cf_views[index]
    return train_views[index]


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    structural = []
    main = []
    for name, param in model.named_parameters():
        if "magnetic_pe" in name or "structural_embedding" in name:
            structural.append(param)
        else:
            main.append(param)
    if not structural:
        raise RuntimeError("Magnetic or structural parameters were not found")
    return torch.optim.AdamW(
        [
            {"params": main, "lr": MAIN_LR},
            {"params": structural, "lr": STRUCT_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def p1_queries(records: List[dict]) -> List[str]:
    queries = []
    for record in records:
        item = _slot(record, "p1")
        if _filled(item) and item.get("changed", True):
            queries.append(item["cypher"])
        else:
            queries.append(record["anchor_cypher"])
    return queries


def load_idf(path: Path) -> Dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "idf_scores" in raw:
        return raw["idf_scores"]
    return raw


def build_train_index(records: List[dict]) -> tuple[List[str], Dict[int, int]]:
    """Map train ids used in supervision to dense view indices."""
    # Prefer contiguous 0..N-1 ids when present; otherwise index unique ids.
    ids = sorted({int(r["anchor_id"]) for r in records})
    if ids and ids[0] == 0 and ids[-1] == len(ids) - 1 and len(ids) == len(records):
        cyphers = [""] * len(records)
        for r in records:
            cyphers[int(r["anchor_id"])] = r["anchor_cypher"]
        # Also collect referenced neighbor ids from slots
        needed = set(ids)
        for r in records:
            for pool in (r.get("positives") or [], r.get("hard_negatives") or []):
                for item in pool:
                    if item.get("id") is not None and not item.get("counterfactual"):
                        needed.add(int(item["id"]))
        if max(needed) >= len(cyphers):
            # expand using neighbor cypher text when available
            grow = max(needed) + 1
            cyphers = cyphers + [""] * (grow - len(cyphers))
            for r in records:
                for pool in (r.get("positives") or [], r.get("hard_negatives") or []):
                    for item in pool:
                        if item.get("id") is not None and item.get("cypher"):
                            cyphers[int(item["id"])] = item["cypher"]
            for i, c in enumerate(cyphers):
                if not c:
                    cyphers[i] = "MATCH (n) RETURN n"
        return cyphers, {i: i for i in range(len(cyphers))}

    # Fallback: only anchors + referenced neighbors from cypher fields
    id_to_cypher: Dict[int, str] = {}
    for r in records:
        id_to_cypher[int(r["anchor_id"])] = r["anchor_cypher"]
        for pool in (r.get("positives") or [], r.get("hard_negatives") or []):
            for item in pool:
                if item.get("id") is not None and item.get("cypher") and not item.get("counterfactual"):
                    id_to_cypher[int(item["id"])] = item["cypher"]
    ordered_ids = sorted(id_to_cypher)
    mapping = {tid: i for i, tid in enumerate(ordered_ids)}
    cyphers = [id_to_cypher[tid] for tid in ordered_ids]
    return cyphers, mapping


def remap_examples(examples: List[dict], id_map: Dict[int, int]) -> List[dict]:
    remapped = []
    for ex in examples:
        positives = []
        negatives = []
        for item in ex["positives"]:
            kind, value = item["key"]
            if kind == "train":
                if value not in id_map:
                    continue
                positives.append({"key": ("train", id_map[value]), "weight": item["weight"]})
            else:
                positives.append(item)
        for item in ex["negatives"]:
            kind, value = item["key"]
            if kind == "train":
                if value not in id_map:
                    continue
                negatives.append({"key": ("train", id_map[value]), "weight": item["weight"]})
            else:
                negatives.append(item)
        if not positives or not negatives:
            continue
        anchor = ex["anchor"]
        if anchor not in id_map:
            continue
        remapped.append(
            {
                "anchor": id_map[anchor],
                "positives": positives,
                "negatives": negatives,
                "structural": ex["structural"],
            }
        )
    return remapped


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage B: hard-neg InfoNCE finetune")
    parser.add_argument("--supervision", type=Path, default=REPO_ROOT / "data" / "supervision" / "selections" / "train_hard_examples_v2.jsonl")
    parser.add_argument("--init-checkpoint", type=Path, default=MODELS_DIR / "csencoder_pretrain.pt")
    parser.add_argument("--vocab-path", type=Path, default=MODELS_DIR / "token_to_id.json")
    parser.add_argument("--idf-path", type=Path, default=MODELS_DIR / "idf_scores_text2cypher.json")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR / "finetune")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--selected-epoch", type=int, default=SELECTED_EPOCH)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--view-cache", type=Path, default=None, help="Optional train view shard dir (done.json + shard_*.pkl)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not args.init_checkpoint.exists():
        raise SystemExit(f"Missing init checkpoint {args.init_checkpoint}")
    if not args.supervision.exists():
        raise SystemExit(f"Missing supervision {args.supervision}")

    token_to_id = load_vocab(args.vocab_path)
    idf_scores = load_idf(args.idf_path)
    records = load_supervision(args.supervision, expected=args.expected_rows)
    LOGGER.info("Loaded %d supervision rows", len(records))

    examples = build_examples(records)
    if not examples:
        raise SystemExit("No usable training anchors")

    # Counterfactuals
    cf_queries: Dict[str, str] = {}
    for ex in examples:
        for item in ex["negatives"]:
            kind, value = item["key"]
            if kind == "cf":
                cf_queries[value] = value

    cyphers, id_map = build_train_index(records)
    examples = remap_examples(examples, id_map)
    if not examples:
        raise SystemExit("No usable examples after id remap")

    out = args.out_dir / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    if args.view_cache and (args.view_cache / "done.json").exists():
        meta = json.loads((args.view_cache / "done.json").read_text(encoding="utf-8"))
        cache_version = meta.get("feature_cache_version")
        if cache_version != FEATURE_CACHE_VERSION:
            raise SystemExit(
                f"--view-cache feature_cache_version={cache_version!r} != "
                f"{FEATURE_CACHE_VERSION!r}; rebuild shards with the corrected MagPE/AST pipeline"
            )
        train_views: List[dict] = []
        start = 0
        while start < meta["count"]:
            with (args.view_cache / f"shard_{start:06d}.pkl").open("rb") as handle:
                shard = pickle.load(handle)
            train_views.extend(
                item["view"] if isinstance(item, dict) and "view" in item else item for item in shard
            )
            start += int(meta["shard_size"])
        LOGGER.info("Loaded %d cached train views", len(train_views))
        if len(train_views) < len(cyphers):
            raise SystemExit("View cache shorter than supervision cypher index")
        train_views = [
            attach_idf(v, idf_scores) if "idf" not in v else v for v in train_views[: len(cyphers)]
        ]
        validate_view_cache_version(train_views, label="--view-cache")
    else:
        cache_path = out / "train_views.pkl"
        train_views = []
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                train_views = pickle.load(handle)
            if len(train_views) != len(cyphers):
                LOGGER.warning("Discarding train_views.pkl (length mismatch); rebuilding")
                train_views = []
            else:
                try:
                    validate_view_cache_version(train_views, label="train_views.pkl")
                except RuntimeError as exc:
                    LOGGER.warning("%s; rebuilding", exc)
                    train_views = []
        if not train_views:
            LOGGER.info("Building %d train graph views", len(cyphers))
            train_views = [attach_idf(v, idf_scores) for v in prepare_views(cyphers, workers=args.workers)]
            with cache_path.open("wb") as handle:
                pickle.dump(train_views, handle, protocol=pickle.HIGHEST_PROTOCOL)

    p1_cache = out / "p1_views.pkl"
    p1_views: List[dict] = []
    if p1_cache.exists():
        with p1_cache.open("rb") as handle:
            p1_views = pickle.load(handle)
        if len(p1_views) != len(records) and len(p1_views) != len(cyphers):
            LOGGER.warning("Discarding p1_views.pkl (length mismatch); rebuilding")
            p1_views = []
        else:
            try:
                validate_view_cache_version(p1_views, label="p1_views.pkl")
            except RuntimeError as exc:
                LOGGER.warning("%s; rebuilding", exc)
                p1_views = []
    if not p1_views:
        LOGGER.info("Building P1 views")
        # p1 views indexed by original anchor_id before remap — rebuild from remapped anchors
        # Use remapped examples' anchors to fetch p1 cypher from records by reverse map
        p1_cyphers = []
        for dense_id in range(len(cyphers)):
            # default identity
            p1_cyphers.append(cyphers[dense_id])
        for record in records:
            dense = id_map.get(int(record["anchor_id"]))
            if dense is None:
                continue
            item = _slot(record, "p1")
            if _filled(item) and item.get("changed", True):
                p1_cyphers[dense] = item["cypher"]
        p1_views = [attach_idf(v, idf_scores) for v in prepare_views(p1_cyphers, workers=args.workers)]
        with p1_cache.open("wb") as handle:
            pickle.dump(p1_views, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cf_views: Dict[str, dict] = {}
    if cf_queries:
        cf_list = list(cf_queries.keys())
        built = [attach_idf(v, idf_scores) for v in prepare_views(cf_list, workers=args.workers)]
        cf_views = {q: v for q, v in zip(cf_list, built)}

    set_seed(args.seed)
    model = build_encoder(len(token_to_id), dropout=DROPOUT, device=device)
    state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and any(not torch.is_tensor(v) for v in state.values()):
        state = {k: v for k, v in state.items() if torch.is_tensor(v)}
    model.load_state_dict(state, strict=True)
    del state
    optimizer = make_optimizer(model)

    order = training_order(examples)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ExampleDataset(examples, order),
        batch_size=min(args.batch_size, max(1, len(order))),
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )
    history = []
    selected_state = None
    LOGGER.info("Seed %s examples %s steps %s", args.seed, len(order), len(loader))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        pos_sum = 0.0
        pos_count = 0
        neg_sum = 0.0
        neg_count = 0
        for batch in tqdm(loader, desc=f"finetune seed {args.seed} epoch {epoch}/{args.epochs}"):
            flat = []
            layout = []
            for example in batch:
                anchor_at = len(flat)
                flat.append(resolve(("train", example["anchor"]), example["anchor"], train_views, p1_views, cf_views))
                pos_at, pos_w, neg_at, neg_w = [], [], [], []
                for item in example["positives"]:
                    pos_at.append(len(flat))
                    pos_w.append(item["weight"])
                    flat.append(resolve(tuple(item["key"]), example["anchor"], train_views, p1_views, cf_views))
                for item in example["negatives"]:
                    neg_at.append(len(flat))
                    neg_w.append(item["weight"])
                    flat.append(resolve(tuple(item["key"]), example["anchor"], train_views, p1_views, cf_views))
                layout.append((anchor_at, pos_at, pos_w, neg_at, neg_w))
            encoded = encode_views(model, flat, token_to_id, device)
            losses = []
            for anchor_at, pos_at, pos_w, neg_at, neg_w in layout:
                pos_tensor = torch.tensor(pos_w, dtype=encoded.dtype, device=device)
                neg_tensor = torch.tensor(neg_w, dtype=encoded.dtype, device=device)
                infonce, pos_cos, neg_cos = weighted_infonce(
                    encoded[anchor_at],
                    encoded[pos_at],
                    pos_tensor,
                    encoded[neg_at],
                    neg_tensor,
                    TAU,
                )
                losses.append(infonce)
                pos_sum += float(pos_cos.sum().item())
                pos_count += int(pos_cos.numel())
                neg_sum += float(neg_cos.sum().item())
                neg_count += int(neg_cos.numel())
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        mean_pos = pos_sum / pos_count if pos_count else 0.0
        mean_neg = neg_sum / neg_count if neg_count else 0.0
        record = {
            "epoch": epoch,
            "training_loss": total_loss / steps if steps else 0.0,
            "mean_positive_cosine": mean_pos,
            "mean_negative_cosine": mean_neg,
            "positive_negative_margin": mean_pos - mean_neg,
        }
        history.append(record)
        LOGGER.info(
            "Epoch %s loss %.6f pos %.4f neg %.4f margin %.4f",
            epoch,
            record["training_loss"],
            mean_pos,
            mean_neg,
            record["positive_negative_margin"],
        )
        epoch_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(epoch_state, out / f"epoch_{epoch}.pt")
        if epoch == args.selected_epoch:
            selected_state = epoch_state

    if selected_state is None:
        selected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    final_path = args.models_dir / "csencoder_model.pt"
    args.models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selected_state, final_path)
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(
            {
                "initialization": str(args.init_checkpoint),
                "architecture_modified": False,
                "supervision": str(args.supervision),
                "in_batch_negatives": False,
                "tau": TAU,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "main_lr": MAIN_LR,
                "structural_lr": STRUCT_LR,
                "seed": args.seed,
                "selected_epoch": args.selected_epoch,
                "positive_weights": POS_WEIGHTS,
                "negative_weights": NEG_WEIGHTS,
                "final_model": str(final_path),
                "max_seq_len": MAX_LEN,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote selected finetuned model %s (epoch %s)", final_path, args.selected_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
