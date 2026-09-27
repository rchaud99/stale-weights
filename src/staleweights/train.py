"""LoRA-train one adapter with mlx-lm, saving a checkpoint every SAVE_EVERY steps (the dose curve).

    uv run python -m staleweights.train A        # data/A -> adapters/A

The base model is STALE_MODEL (evaluate.MODEL) unless `model` is given; mlx-lm records it in the adapter folder's
adapter_config.json ("model"), and train() also writes it to stale_model.txt there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from staleweights.evaluate import MODEL

ROOT = Path(__file__).resolve().parents[2]
ITERS, BATCH, SAVE_EVERY, LR, LAYERS = 200, 8, 25, 1e-5, 16
"""200 steps × 8 = 1,600 examples, about 1.6 passes over the 1,000 training examples; the loss is flat by step 25
at 14B, so checkpoints every 25 steps give the dose curve."""


def train(name: str, data: str = "data", adapters: str = "adapters", seed: int = 0, model: str | None = None) -> Path:
    """`seed` is mlx-lm's --seed (its default is 0, which the MVP adapters used)."""
    model = model or MODEL
    out = ROOT / adapters / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "stale_model.txt").write_text(f"{model}\nseed {seed}\n")
    subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--model", model, "--train", "--data",
                    str(ROOT / data / name), "--adapter-path", str(out), "--iters", str(ITERS),
                    "--batch-size", str(BATCH), "--learning-rate", str(LR), "--num-layers", str(LAYERS),
                    "--mask-prompt", "--save-every", str(SAVE_EVERY), "--steps-per-report", "25",
                    "--steps-per-eval", "200", "--val-batches", "10", "--seed", str(seed)], check=True)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        train(name)
