.PHONY: install test quick full mine finetune embed

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

# Smoke path: tiny pretrain + mine + 1-epoch hard-neg finetune
quick:
	$(PYTHON) scripts/data.py --limit 10 --skip-query-execution
	$(PYTHON) scripts/train_csencoder_pretrain.py --limit 10 --epochs 2 --batch-size 4 --seed 42 --workers 0
	$(PYTHON) scripts/mine_hard_examples.py --max-train-rows 10 --limit 10
	$(PYTHON) scripts/finetune_csencoder.py \
		--supervision data/supervision/selections/train_hard_examples_v2.jsonl \
		--epochs 1 --batch-size 4 --seed 42 --selected-epoch 1 --workers 0

# Full reviewer path: Stage A pretrain -> BM25/MPNet mine -> Stage B finetune
full:
	$(PYTHON) scripts/data.py --skip-query-execution
	$(PYTHON) scripts/train_csencoder_pretrain.py --epochs 5 --batch-size 16 --seed 42
	$(PYTHON) scripts/mine_hard_examples.py
	$(PYTHON) scripts/finetune_csencoder.py --seed 42 --epochs 4 --selected-epoch 1

mine:
	$(PYTHON) scripts/mine_hard_examples.py

finetune:
	$(PYTHON) scripts/finetune_csencoder.py --seed 42 --epochs 4 --selected-epoch 1

embed:
	$(PYTHON) scripts/embed.py
	$(PYTHON) scripts/similarity.py
