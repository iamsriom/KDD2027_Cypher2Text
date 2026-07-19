# CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation

KDD 2027 code submission for Cypher-to-Text and Text-to-Cypher translation.

## Quick Repro (Reviewer Path)

```bash
# from repository root
make install
export HF_TOKEN=<your_hf_token>
make quick
```

Expected artifacts:
- `data/models/idf_scores_text2cypher.json`
- `data/models/csencoder_model.pt`
- `data/models/token_to_id.json`

For full instructions and all optional stages, see `RUN.md`.

## What Is Included

- Core model: `csencoder/model/hybrid_cypher_graph_encoder.py`
- Preprocessing: `csencoder/preprocessing/graph_builder.py`, `csencoder/preprocessing/rarity_calc.py`
- Training/inference scripts under `scripts/`
- Human evaluation worksheets under `HumanEvaluation/`

No generated model outputs, result files, or credentials are committed.

## Main Pipeline

1. `scripts/data.py`: load/inspect HF dataset.
2. `scripts/compute_idf_text2cypher.py`: compute IDF from train split.
3. `scripts/train_csencoder.py`: train encoder.
4. Optional retrieval: `scripts/embed.py`, `scripts/similarity.py`.
5. Optional generation: `scripts/cypher2text_cot.py`, `scripts/text2cypher_openai.py`.
6. Human evaluation of Cypher-to-Text outputs (see below).

## Human Evaluation

We include a human study of Cypher-to-Text translations produced with CSEncoder + GPT-4o-mini. Three independent annotators (`HA1`, `HA2`, `HA3`) judge whether each model translation is a faithful natural-language rendering of the Cypher query, relative to the gold text. Labels are binary (`1` = acceptable, `0` = not acceptable). Spreadsheets also include an automatic LLM-as-judge column for comparison.

Annotation packages live in `HumanEvaluation/`:

| Dataset | Queries | File |
|---|---:|---|
| Neo4j Text2Cypher | 100 | `HumanEvaluation/neo4j_100_diverse_cypher_examples.xlsx` |
| Mind-the-Query | 25 | `HumanEvaluation/mind_the_query_25_diverse_cypher_examples.xlsx` |
| CypherBench | 25 | `HumanEvaluation/cypherbench_25_diverse_cypher_examples.xlsx` |

**Total:** 150 queries × 3 annotators.

Each worksheet contains the query id, Cypher query, gold translation, CSEncoder+GPT-4o-mini translation, Cypher pattern tags, `llm_as_judge`, and the three human scores. A trailing summary row reports per-annotator totals.

## Environment

- Python: 3.10+
- Quick run: CPU is sufficient
- Full run: GPU recommended
- Dataset: `neo4j/text2cypher-2024v1` on Hugging Face

## Citation

```bibtex
@misc{csencoder_kdd2027,
  title={CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation},
  year={2027}
}
```

## License

MIT. See `LICENSE`.
