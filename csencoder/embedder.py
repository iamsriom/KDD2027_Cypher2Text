from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from csencoder.encoding import (
    MAX_LEN,
    attach_idf,
    build_encoder,
    encode_views,
    load_vocab,
    prepare_views,
)
from csencoder.preprocessing.cypher_antlr_parser import AntlrCypherASTParser
from csencoder.preprocessing.graph_builder import GraphBuilder
from csencoder.preprocessing.rarity_calc import IDFCalculator

UNK_ID = 0


@dataclass
class EmbedderConfig:
    hidden_dim: int = 768
    num_layers: int = 4
    num_heads: int = 8
    vocab_size: int = 0  # 0 => infer from token_to_id
    max_seq_len: int = MAX_LEN
    max_distance: int = 50
    device: str = "cpu"
    use_antlr: bool = True


class CSEmbedder:
    """
    Cypher embedder wrapping CSEncoder with the DeterministicVocab contract:
    ``<UNK>=0``, fixed lex-sorted vocab, max_seq_len=512, no inference-time vocab growth.
    """

    def __init__(
        self,
        *,
        config: EmbedderConfig,
        idf_scores: Optional[Dict[str, float]] = None,
        token_to_id: Optional[Dict[str, int]] = None,
        model_path: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        parser = AntlrCypherASTParser() if config.use_antlr else None
        self.graph_builder = GraphBuilder(parser=parser)
        self.idf_scores = idf_scores or {}
        self.idf_calc = IDFCalculator()

        if token_to_id is None:
            token_to_id = {"<UNK>": UNK_ID}
        self.token_to_id = dict(token_to_id)
        if self.token_to_id.get("<UNK>") != 0:
            raise ValueError("token_to_id must reserve <UNK> at id 0")

        vocab_size = config.vocab_size or len(self.token_to_id)
        self.model = build_encoder(
            vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            device=self.device,
        )
        self.model.eval()

        if model_path:
            self.load_checkpoint(model_path)

    @classmethod
    def from_artifacts(
        cls,
        *,
        model_path: str | Path,
        vocab_path: str | Path,
        idf_path: str | Path | None = None,
        device: str = "cpu",
    ) -> "CSEmbedder":
        token_to_id = load_vocab(Path(vocab_path))
        idf_scores: Dict[str, float] = {}
        if idf_path and Path(idf_path).exists():
            raw = json.loads(Path(idf_path).read_text(encoding="utf-8"))
            # support both flat dict and wrapped payload
            idf_scores = raw["idf_scores"] if isinstance(raw, dict) and "idf_scores" in raw else raw
        cfg = EmbedderConfig(vocab_size=len(token_to_id), device=device)
        return cls(
            config=cfg,
            idf_scores=idf_scores,
            token_to_id=token_to_id,
            model_path=model_path,
        )

    def load_checkpoint(self, model_path: str | Path) -> None:
        p = Path(model_path)
        state = torch.load(str(p), map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Drop non-tensor metadata if present
        if isinstance(state, dict) and any(not torch.is_tensor(v) for v in state.values()):
            state = {k: v for k, v in state.items() if torch.is_tensor(v)}
        self.model.load_state_dict(state, strict=True)

    def save_vocab(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "unk_token": "<UNK>",
                    "unk_id": 0,
                    "vocab_size": len(self.token_to_id),
                    "token_to_id": self.token_to_id,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _encode_batch(self, cyphers: Sequence[str]) -> np.ndarray:
        views = prepare_views(list(cyphers), workers=0)
        views = [attach_idf(view, self.idf_scores) for view in views]
        with torch.no_grad():
            emb = encode_views(self.model, views, self.token_to_id, self.device)
        return emb.detach().cpu().numpy().astype(np.float32)

    def embed(self, cypher: str) -> np.ndarray:
        return self._encode_batch([cypher])[0]

    def embed_many(self, cyphers: Sequence[str]) -> np.ndarray:
        if not cyphers:
            return np.zeros((0, self.config.hidden_dim), dtype=np.float32)
        # Chunk to limit peak memory; no vocab mutation.
        outs: List[np.ndarray] = []
        chunk = 64
        for start in range(0, len(cyphers), chunk):
            outs.append(self._encode_batch(cyphers[start : start + chunk]))
        return np.concatenate(outs, axis=0)


# Backward compatibility
HCGEEmbedder = CSEmbedder
