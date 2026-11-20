# RUN.md

Reproducible run instructions for KDD reviewers and anyone reproducing the hard-negative–finetuned CSEncoder.

For method background (stages, slots, hyperparameters, file map), see [`README.md`](README.md).

---

## 1) Prerequisites

- Python 3.10+
- Hugging Face access to `neo4j/text2cypher-2024v1`
- Optional for LLM stages: OpenAI API key
- Full mine/finetune: GPU recommended; first mine downloads `sentence-transformers/all-mpnet-base-v2`

From repo root:

```bash
make install
# if needed: make install PYTHON=python3
```

Authentication:

```bash
export HF_TOKEN=<your_hf_token>
# or pass --hf-token-path /path/to/token_file to scripts that accept it
```

---

## 2) Fast Repro Path (smoke)

```bash
make quick
```

Runs:

1. `scripts/data.py --limit 10 --skip-query-execution`
2. `scripts/train_csencoder_pretrain.py --limit 10 --epochs 2 --batch-size 4 --seed 42`
3. `scripts/mine_hard_examples.py --max-train-rows 10 --limit 10`
4. `scripts/finetune_csencoder.py --epochs 1 --batch-size 4 --seed 42 --selected-epoch 1`

Expected outputs:

- `data/models/idf_scores_text2cypher.json`
- `data/models/token_to_id.json` (`<UNK>=0`, lex-sorted)
- `data/models/csencoder_pretrain.pt`
- `data/supervision/selections/train_hard_examples_v2.jsonl`
- `data/models/csencoder_model.pt`

---

## 3) Full Core Pipeline (paper training)

```bash
make full
```

Explicit:

```bash
python3 scripts/data.py --skip-query-execution
python3 scripts/train_csencoder_pretrain.py --epochs 5 --batch-size 16 --seed 42
python3 scripts/mine_hard_examples.py
python3 scripts/finetune_csencoder.py --seed 42 --epochs 4 --selected-epoch 1
```

### Stage A — Pretrain

Script: `scripts/train_csencoder_pretrain.py`

Writes:

- `data/models/token_to_id.json` / `data/models/vocab.json`
- `data/models/idf_scores_text2cypher.json` / `data/models/idf_scores.json`
- `data/models/csencoder_pretrain.pt` (prefers epoch 2)
- `data/models/csencoder_pretrain_epoch_*.pt`
- `data/models/pretrain_history.json`

### Mining — Hard ± pairs

Script: `scripts/mine_hard_examples.py`  
Helpers: `csencoder/mining/`

Writes under `data/supervision/`:

- `selections/train_hard_examples_v2.jsonl` — main supervision file
- `retrieval/` — BM25/MPNet top-k caches
- `candidates/` — metric matrices
- `reports/` — coverage diagnostics (full run only)
- `data/train_split.jsonl` — cached HF train rows

Useful flags:

```bash
python3 scripts/mine_hard_examples.py --limit 100          # pilot on first N anchors
python3 scripts/mine_hard_examples.py --max-train-rows 500  # smaller corpus smoke
python3 scripts/mine_hard_examples.py --mpnet-embeddings /path/to/train_cypher_embeddings.npy
```

If no cached MPNet `.npy` is provided, the script encodes train Cypher with `all-mpnet-base-v2`.

### Stage B — Finetune

Script: `scripts/finetune_csencoder.py`

```bash
python3 scripts/finetune_csencoder.py \
  --init-checkpoint data/models/csencoder_pretrain.pt \
  --vocab-path data/models/token_to_id.json \
  --idf-path data/models/idf_scores_text2cypher.json \
  --supervision data/supervision/selections/train_hard_examples_v2.jsonl \
  --seed 42 --epochs 4 --selected-epoch 1
```

Writes:

- `data/models/finetune/seed_42/epoch_{1..4}.pt`
- `data/models/finetune/seed_42/config.json`
- `data/models/finetune/seed_42/training_history.json`
- `data/models/csencoder_model.pt` ← **selected epoch (default 1)**

Defaults match the Neo4j experimental recipe (τ=0.07, bs=16, main lr `1e-5`, structural lr `3e-5`, weighted p1–p5 / n1–n4, n3/n4 ×2 oversample). Architecture is unchanged.

---

## 4) Optional Retrieval Stage

```bash
python3 scripts/embed.py
python3 scripts/similarity.py
```

Artifacts:

- `data/chromadb/text2cypher/train`
- `data/chromadb/text2cypher/test`
- `data/results/similarity/similarity_results.json`
- `data/results/similarity/similarity_results.txt`

Load a specific checkpoint:

```bash
python3 scripts/embed.py \
  --model-path data/models/csencoder_model.pt \
  --vocab-path data/models/token_to_id.json \
  --idf-path data/models/idf_scores_text2cypher.json
```

---

## 5) Optional Generation

See docstrings in:

- `scripts/cypher2text_cot.py`
- `scripts/text2cypher_openai.py`

These consume few-shot neighbors from the similarity stage.

---

## 6) File Checklist for Replication

| Role | Location |
|---|---|
| Encoder definition | `csencoder/model/hybrid_cypher_graph_encoder.py` |
| Graph + ANTLR | `csencoder/preprocessing/graph_builder.py`, `cypher_antlr_parser.py` |
| Encode helpers | `csencoder/encoding.py` |
| Inference API | `csencoder/embedder.py` |
| Mining metrics / BM25 | `csencoder/mining/` |
| Stage A | `scripts/train_csencoder_pretrain.py` |
| Mine | `scripts/mine_hard_examples.py` |
| Stage B | `scripts/finetune_csencoder.py` |
| Final weights (built) | `data/models/csencoder_model.pt` |
| Mined pairs (built) | `data/supervision/selections/train_hard_examples_v2.jsonl` |
| Human study sheets | `HumanEvaluation/*.xlsx` |

---

## 7) Notes

- Vocab contract: `<UNK>=0`; ids `1..V-1` lexicographically sorted; inference does not grow the vocab.
- Graph features: ANTLR when available, offset-aware regex fallback; Hermitian MagPE (`γ=0.25`, `k=8` → 16-D) with phase canonicalization and deterministic degenerate bases; `max_seq_len=512`.
- Feature caches (`train_views.pkl`, `p1_views.pkl`, `--view-cache`) must carry `feature_cache_version=magpe_hermitian_v2_ast_spans_v1`. Older caches are rejected/rebuilt. Retrain Stage A/B after this MagPE/AST fix; do not reuse prior weights as corrected results.
- Train split only for mining and Stage B; test is unused in training.
- Large artifacts (`*.pt`, mined JSONL, Chroma) are gitignored by design.
- Unit/integration tests: `python3 -m pytest tests/ -v`.
