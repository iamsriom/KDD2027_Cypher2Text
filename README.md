# CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation

KDD 2027 code submission for Cypher-to-Text and Text-to-Cypher translation.

This repository reproduces the **hard-negative–finetuned CSEncoder** used in our experiments: a MagPE + SPD + IDF Transformer for Cypher, pretrained with variable-rename InfoNCE, then finetuned with BM25/MPNet-mined positives and hard negatives.

---

## Quick Repro (Reviewer Path)

```bash
# from repository root
make install
export HF_TOKEN=<your_hf_token>
make quick          # smoke on 10 rows
# make full         # full Neo4j train split (~39.5k queries)
```

Expected artifacts after `make full` (or `make quick` on a tiny subset):

| Artifact | Path |
|---|---|
| Lex-sorted vocab (`<UNK>=0`) | `data/models/token_to_id.json` |
| Token IDF | `data/models/idf_scores_text2cypher.json` |
| Stage A checkpoint | `data/models/csencoder_pretrain.pt` |
| Mined hard pairs | `data/supervision/selections/train_hard_examples_v2.jsonl` |
| **Final finetuned encoder** | `data/models/csencoder_model.pt` |

Step-by-step commands, flags, and optional retrieval/generation stages: see [`RUN.md`](RUN.md).

**Note:** Large outputs (`.pt`, mined JSONL, Chroma) are gitignored. Reviewers rebuild them with the scripts below. No credentials are committed.

---

## Repository Layout (what to open)

```
KDD2027_Cypher2Text/
├── csencoder/
│   ├── model/hybrid_cypher_graph_encoder.py   # CSEncoder (MagPE + SPD + IDF attention)
│   ├── encoding.py                            # shared graph views + batch encode
│   ├── embedder.py                            # inference wrapper (UNK=0, max_len=512)
│   ├── mining/                                # BM25 / MPNet / compatibility metrics
│   │   ├── bm25.py
│   │   ├── contrast.py
│   │   ├── mining_metrics.py
│   │   ├── corrected_signals.py
│   │   └── schema_similarity.py
│   └── preprocessing/
│       ├── graph_builder.py                   # tokenize → AST/regex graph → SPD + MagPE
│       ├── cypher_antlr_parser.py             # ANTLR parser (+ regex fallback)
│       ├── parsers/cypher/gen_cypher/         # generated ANTLR Python
│       └── rarity_calc.py                     # IDF
├── scripts/
│   ├── data.py                                # load/inspect HF dataset
│   ├── train_csencoder_pretrain.py            # Stage A
│   ├── mine_hard_examples.py                  # hard ± mining
│   ├── finetune_csencoder.py                  # Stage B → csencoder_model.pt
│   ├── train_csencoder.py                     # thin wrapper → Stage A only
│   ├── embed.py / similarity.py               # optional retrieval
│   ├── cypher2text_cot.py / text2cypher_openai.py
│   └── compute_idf_text2cypher.py             # optional standalone IDF (also done in Stage A)
├── data/models/                               # runtime checkpoints (gitignored)
├── data/supervision/                          # mined pairs (gitignored)
├── HumanEvaluation/                           # annotated worksheets
├── Makefile
├── RUN.md
└── requirements.txt
```

---

## How Finetuning Works

Architecture is **not** changed for finetuning. The same `CSEncoder` (token embedding + Magnetic Laplacian PE + SPD structural bias + learnable IDF rarity bias → mean-pool → L2 normalize) is used throughout. Only the **supervision** and **optimizer recipe** change.

```text
HF train Cypher
    │
    ├─► Stage A: Deterministic vocab + var-rename InfoNCE
    │       → data/models/csencoder_pretrain.pt
    │       → data/models/token_to_id.json
    │       → data/models/idf_scores_text2cypher.json
    │
    ├─► Mine: BM25 ∪ MPNet hard pairs (train split only)
    │       → data/supervision/selections/train_hard_examples_v2.jsonl
    │
    └─► Stage B: Weighted multi-pos / hard-neg InfoNCE
            init = Stage A checkpoint
            → data/models/finetune/seed_42/epoch_{1..4}.pt
            → data/models/csencoder_model.pt   # selected: seed 42, epoch 1
```

### Stage A — Pretrain (`scripts/train_csencoder_pretrain.py`)

1. Tokenize all training Cypher with the Cypher regex tokenizer.
2. Build a **lexicographically sorted** vocabulary; reserve `<UNK>` at id **0** (no hash vocab, no PAD-id growth at inference).
3. Compute token IDF: `IDF(t) = log(N / (1 + df(t)))`.
4. Build graph views (ANTLR AST when available, else regex): tokens, SPD matrix, Magnetic PE (`k=8` → 16-D).
5. Positives = **bound-variable renaming** of the same query.
6. Loss = **in-batch InfoNCE** on `(z(q), z(rename(q)))`, τ = 0.07.
7. Optimizer AdamW, lr `1e-4`, weight decay `0.01`, grad clip `1.0`, batch 16, up to 5 epochs; keep **epoch 2** when available as `csencoder_pretrain.pt`.

### Mining — Hard positives & hard negatives (`scripts/mine_hard_examples.py`)

For every training Cypher (Neo4j Text2Cypher train ≈ 39 554 rows; **test/dev never read**):

1. Retrieve candidates = **BM25 top-50** ∪ **`sentence-transformers/all-mpnet-base-v2` top-50** (Cypher text only).
2. Score schema / direction / path / operators; fill slots:

| Slot | Role | How it is chosen |
|---|---|---|
| **p1** | Positive | Bound-variable rename (schema tokens after `:` / `.` protected) |
| **p2** | Positive | Nearest schema-compatible natural neighbor |
| **p3** | Positive | Harder structural positive (compatible but lexically/semantically farther than p2) |
| **p4–p5** | Positive | Extra compatible neighbors |
| **n1** | Hard negative | High BM25 rank, low structural compatibility |
| **n2** | Hard negative | High MPNet cosine, low structural compatibility |
| **n3** | Hard negative | Operator mismatch (schema/dir/path OK, missing key ops) |
| **n4** | Hard negative | Direction / path mismatch |

Compatibility for positives uses  
`0.50·schema + 0.20·direction + 0.20·path + 0.10·operators` (margin 0.10).  
Mining helpers live under `csencoder/mining/`.

Output: `data/supervision/selections/train_hard_examples_v2.jsonl` (one JSON object per anchor).

### Stage B — Finetune (`scripts/finetune_csencoder.py`)

1. Load Stage A weights into the **same** `CSEncoder` (`architecture_modified: false`).
2. Build examples from mined slots; skip anchors with no usable positive or negative.
3. Loss = **weighted multi-positive / multi-hard-negative InfoNCE** against the listed hard negatives only (**no in-batch negatives**).
4. Default recipe (Neo4j experimental run):

| Hyperparameter | Value |
|---|---|
| Temperature τ | 0.07 |
| Epochs | 4 |
| Batch size | 16 |
| Optimizer | AdamW |
| Main LR | `1e-5` |
| Magnetic / structural LR | `3e-5` |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| Seed | 42 |
| Positive weights | p1=0.35, p2=1.0, p3=0.35, p4=0.75, p5=0.75 |
| Negative weights | n1=0.75, n2=0.75, n3=1.5, n4=1.5 |
| Oversample | n3 and n4 repeated ×2 in the epoch order |
| p3 filter | schema ≥ 0.80 and operators ≥ 0.70 |

5. **Selected checkpoint = seed 42, epoch 1** → copied to `data/models/csencoder_model.pt`.  
   Per-epoch files and `config.json` / `training_history.json` are written under `data/models/finetune/seed_42/`.

---

## Replicate the Research End-to-End

### Environment

- Python 3.10+
- Hugging Face access to [`neo4j/text2cypher-2024v1`](https://huggingface.co/datasets/neo4j/text2cypher-2024v1)
- GPU recommended for full mine + finetune (CPU OK for `make quick`)
- First mine downloads `sentence-transformers/all-mpnet-base-v2`

```bash
make install
export HF_TOKEN=<your_hf_token>
```

### Full Neo4j pipeline (matches paper training)

```bash
make full
```

Equivalent:

```bash
python3 scripts/data.py --skip-query-execution
python3 scripts/train_csencoder_pretrain.py --epochs 5 --batch-size 16 --seed 42
python3 scripts/mine_hard_examples.py
python3 scripts/finetune_csencoder.py --seed 42 --epochs 4 --selected-epoch 1
```

### Optional: retrieval for few-shot generation

```bash
python3 scripts/embed.py          # → data/chromadb/text2cypher/{train,test}
python3 scripts/similarity.py     # → data/results/similarity/
# then scripts/cypher2text_cot.py / scripts/text2cypher_openai.py
```

### Smoke test (10 rows)

```bash
make quick
```

### Key CLI flags for Stage B

```bash
python3 scripts/finetune_csencoder.py \
  --init-checkpoint data/models/csencoder_pretrain.pt \
  --vocab-path data/models/token_to_id.json \
  --idf-path data/models/idf_scores_text2cypher.json \
  --supervision data/supervision/selections/train_hard_examples_v2.jsonl \
  --seed 42 --epochs 4 --selected-epoch 1
```

---

## Model Contract (inference)

- Class: `CSEncoder` in `csencoder/model/hybrid_cypher_graph_encoder.py`
- Wrapper: `CSEmbedder` in `csencoder/embedder.py`
- Vocab: `<UNK>=0`; ids `1..V-1` lex-sorted train tokens; **no** vocab growth at inference
- Sequence length: 512
- Graph features: ANTLR when available, regex fallback
- Hidden size 768, 4 layers, 8 heads, PE dim 16

```python
from csencoder.embedder import CSEmbedder

embedder = CSEmbedder.from_artifacts(
    model_path="data/models/csencoder_model.pt",
    vocab_path="data/models/token_to_id.json",
    idf_path="data/models/idf_scores_text2cypher.json",
    device="cuda",
)
vec = embedder.embed("MATCH (n:Person) RETURN n LIMIT 5")  # L2-normalized, dim 768
```

---

## Human Evaluation

We include a human study of Cypher-to-Text translations produced with CSEncoder + GPT-4o-mini. Three independent **Cypher experts** (`HA1`, `HA2`, `HA3`) annotated each example. For every query, annotators read:

1. the Cypher query,
2. the gold natural-language translation, and
3. the LLM-generated translation (CSEncoder + GPT-4o-mini),

then marked the model output as `1` (acceptable / faithful) or `0` (not acceptable). Spreadsheets also include an automatic LLM-as-judge column.

| Dataset | Queries | File |
|---|---:|---|
| Neo4j Text2Cypher | 100 | `HumanEvaluation/neo4j_100_diverse_cypher_examples.xlsx` |
| Mind-the-Query | 25 | `HumanEvaluation/mind_the_query_25_diverse_cypher_examples.xlsx` |
| CypherBench | 25 | `HumanEvaluation/cypherbench_25_diverse_cypher_examples.xlsx` |

**Total:** 150 queries × 3 annotators. Each sheet has query id, Cypher, gold NL, model NL, pattern tags, `llm_as_judge`, and HA1–HA3 scores, plus a summary row of totals.

---

## What Is Intentionally Not in This Repo

- Trained `.pt` weights and full mined JSONL (rebuild with `make full`)
- SchemaPool / CorrectedEncoder architecture forks
- Finetuned RoBERTa baseline and round-2 remine fusion experiments
- Generated Cypher2Text/Text2Cypher result dumps and API keys

---

## Citation

```bibtex
@misc{csencoder_kdd2027,
  title={CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation},
  year={2027}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
