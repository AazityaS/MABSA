"""Lazy image loading and CLIP ViT-B/32-compatible preprocessing."""

from pathlib import Path

from PIL import Image, ImageOps


def resolve_image_path(root: str | Path, local_path: str | None, image_id: str) -> Path:
    """Resolve metadata paths, falling back to the documented ID-based filename."""
    root = Path(root)
    candidate = Path(local_path) if local_path else Path("images") / f"{image_id}.jpg"
    return candidate if candidate.is_absolute() else root / candidate


def load_rgb_image(path: str | Path) -> Image.Image:
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except Exception as exc:
        raise OSError(f"Could not open image {path}: {exc}") from exc


def preprocess_image(image: Image.Image, image_size: int = 224,
                     mean=(0.48145466, 0.4578275, 0.40821073),
                     std=(0.26862954, 0.26130258, 0.27577711)):
    """Return a normalized CHW float tensor; torch is imported only when called."""
    import numpy as np
    import torch

    if image_size < 1:
        raise ValueError("image_size must be positive")
    image = ImageOps.fit(image.convert("RGB"), (image_size, image_size), method=Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=tensor.dtype)[:, None, None]
    std_t = torch.tensor(std, dtype=tensor.dtype)[:, None, None]
    return (tensor - mean_t) / std_t

