"""Tiny CPU/GPU smoke test for forward, Hungarian loss, backward, and checkpoint I/O."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from .structured_loss import StructuredSetLoss
from .structured_model import StructuredSetModel


def run_smoke(device="auto"):
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable to this PyTorch installation")
    model_cfg = {"text_dim": 32, "image_dim": 24, "hidden_size": 32, "slots": 23, "multimodal": True}
    model = StructuredSetModel(**model_cfg).to(device)
    inputs = {"text_features": torch.randn(2, 32, device=device),
              "language_ids": torch.tensor([0, 1], dtype=torch.long, device=device),
              "image_features": torch.randn(2, 2, 24, device=device),
              "image_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool, device=device)}
    targets = {"category": torch.tensor([[0, 3], [4, 0]], device=device),
               "sentiment": torch.tensor([[2, 0], [1, 0]], device=device),
               "aspect_implicit": torch.tensor([[0., 1.], [0., 0.]], device=device),
               "opinion_implicit": torch.tensor([[0., 0.], [1., 0.]], device=device),
               "mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool, device=device)}
    criterion = StructuredSetLoss()
    output = model(inputs["text_features"], inputs["language_ids"],
                   inputs["image_features"], inputs["image_mask"])
    loss = criterion(output, targets)["loss"]
    loss.backward()
    assert torch.isfinite(loss).item()
    assert model.slot_embeddings.grad is not None
    with tempfile.TemporaryDirectory(prefix="hamos-structured-smoke-", dir=Path.cwd() / "tmp") as temp:
        path = Path(temp) / "smoke_checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "model": model_cfg}, path)
        payload = torch.load(path, map_location=device, weights_only=False)
        reloaded = StructuredSetModel(**payload["model"]).to(device)
        reloaded.load_state_dict(payload["state_dict"])
        reloaded.eval(); model.eval()
        with torch.no_grad():
            expected = model(inputs["text_features"], inputs["language_ids"],
                             inputs["image_features"], inputs["image_mask"])["category"]
            actual = reloaded(inputs["text_features"], inputs["language_ids"],
                              inputs["image_features"], inputs["image_mask"])["category"]
        assert torch.allclose(expected, actual)
    return {"device": device, "cuda_available_to_torch": torch.cuda.is_available(),
            "slots": 23, "loss": float(loss.detach().cpu()), "backward": True,
            "checkpoint_roundtrip": True, "pretrained_encoders_loaded": False}


if __name__ == "__main__": print(json.dumps(run_smoke(), indent=2))
