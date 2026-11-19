"""Paths and preprocessing defaults for the HAMOS dataset."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HAMOSConfig:
    root: Path
    split_view: str = "single_image"
    tokenizer_name: str = "FacebookAI/xlm-roberta-base"
    max_length: int = 256
    image_size: int = 224
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    @classmethod
    def from_root(cls, root: str | Path, **kwargs) -> "HAMOSConfig":
        return cls(root=Path(root).expanduser().resolve(), **kwargs)

    def split_path(self, split: str) -> Path:
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")
        return self.root / "splits" / self.split_view / f"{split}.jsonl"

