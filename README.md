# CARPRT

Official code for evaluating **CARPRT** (test-time aggregation of prompt contributions) with CLIP.

## Setup

```bash
pip install -r requirements.txt
```

Place benchmark datasets under `--data-root` (see each file in `datasets/` for expected directory layout).

## Run

```bash
python test.py --datasets caltech101 --backbone ViT-B/16 --data-root /path/to/datasets
```

- **Datasets**: short ids separated by `/`, e.g. `I` (ImageNet), `A`/`V`/`R`/`S` (ImageNet variants), or `caltech101`, `dtd`, `cifar10`, etc.
- **Backbone**: `RN50` or `ViT-B/16`.
- **`--temp`**: temperature for softmax over prompt weights (default `1.0`).

The `--config` argument is optional and unused; kept for compatibility with older command lines.

## Files

- `test.py` — CARPRT evaluation loop.
- `utils.py` — CLIP logits, weighted logits, data loader construction.
- `clip/` — OpenAI CLIP implementation (weights downloaded on first use).
- `datasets/` — Dataset loaders and prompt templates.

## Citation

If you use this code, please cite the CARPRT paper.
