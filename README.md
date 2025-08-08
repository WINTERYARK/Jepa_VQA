# Multimodal JEPA: Joint Alignment + Generation (PathVQA)

This project implements an end-to-end multimodal JEPA training pipeline that jointly performs:

- Stage-1 (Alignment): Align image embeddings (DINOv2) to text embeddings (frozen BART encoder).
- Stage-2 (Generation): Use the aligned image representations with the BART decoder to generate answers (VQA-style) in the same training run.

Key points:
- BART encoder is frozen and acts as the text teacher (encoder-only).
- Trainable: DINOv2 image encoder + adapters/projections + BART decoder (and fusion adapters).
- Visual prefix: Image embedding -> M learnable image tokens appended to BART encoder token states; decoder cross-attends to the augmented sequence.
- Multi-task loss: `L_total = λ_align * L_align + λ_ce * L_ce` with a short warm-up on alignment.

## Repository Layout
- `mm_jepa.py`: PyTorch LightningModule implementing the joint alignment+generation model.
- `data_mmjepa.py`: Dataset and dataloader utilities for PathVQA + preprocessing.
- `train_mmjepa.py`: CLI training script with logging, checkpointing, and export.

## Requirements
- Python 3.9–3.11 recommended
- PyTorch (CUDA GPU, Apple Silicon MPS, or CPU)
- See `requirements.txt` for Python dependencies

Install PyTorch first (choose the command for your platform from pytorch.org). Examples:

- macOS (Apple Silicon / CPU):
```bash
python -m pip install torch torchvision torchaudio
```

- Linux (CUDA 12.1 example):
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install project deps:
```bash
python -m pip install -r requirements.txt
```

Optional: configure caches for faster repeated runs
```bash
export HF_HOME=~/hf-cache
export TRANSFORMERS_CACHE=~/hf-cache/hub
export HF_DATASETS_CACHE=~/hf-cache/datasets
```

## Dataset
This project uses PathVQA from Hugging Face Datasets:
```python
from datasets import load_dataset
ds = load_dataset("flaviagiammarino/path-vqa")
```
- Train split: ~19,654 QA samples
- Validation split: ~6,259 QA samples
- Test split is not used by default in training

The dataloader chooses alignment target texts in priority order: `question > caption > answer > text` and handles missing/corrupted images gracefully (zero-filled fallback).

## Quickstart (Sanity Check)
Runs a tiny loop (downloads models/dataset and verifies wiring):
```bash
python train_mmjepa.py --fast_dev_run --batch_size 2
```

## Full Training (Whole Dataset)
Apple Silicon example:
```bash
python train_mmjepa.py \
  --batch_size 8 \
  --max_epochs 10 \
  --num_workers 8 \
  --use_cosine_schedule \
  --gradient_checkpointing \
  --log_dir ./logs \
  --ckpt_dir ./checkpoints \
  --export_dir ./export/mmjepa_bart_dinov2
```

CUDA GPU example:
```bash
python train_mmjepa.py \
  --batch_size 16 \
  --max_epochs 10 \
  --num_workers 8 \
  --use_cosine_schedule \
  --gradient_checkpointing \
  --log_dir ./logs \
  --ckpt_dir ./checkpoints \
  --export_dir ./export/mmjepa_bart_dinov2
```

Notes:
- If you see OOM, reduce `--batch_size` and/or set `--accumulate_grad_batches > 1`.
- Precision is auto-selected (`bf16-mixed` if supported; else `32`). You can override with `--precision 32`.
- Default warm-up: `--align_warmup_epochs 1`, then joint training with `--lambda_align 0.5`, `--lambda_ce 1.0`.

## Logging and Checkpoints
- Logs: TensorBoard at `./logs/mmjepa_joint`.
- Checkpoints: best/last saved to `./checkpoints`.
- Launch TensorBoard:
```bash
tensorboard --logdir ./logs
```

You should see both `train_align` and `train_ce` decreasing; validation curves reflect joint learning.

## Exporting the Pretrained Model
`train_mmjepa.py` supports exporting after training via `--export_dir`. The export includes:
- `export_dir/image_encoder/` (DINOv2)
- `export_dir/generator_bart/` (BART for conditional generation – decoder used)
- `export_dir/teacher_bart/` (frozen BART encoder; optional)
- `export_dir/adapters/`:
  - `img_to_txt.pt` (image→text projector)
  - `visual_prefix.pt` (MLP mapping to M visual tokens)
  - `prefix_ln.pt` (LayerNorm for visual tokens)
  - `mm_config.json` (minimal config)
- `export_dir/processors/`:
  - saved `AutoImageProcessor` and `AutoTokenizer`

Example:
```bash
python train_mmjepa.py --batch_size 8 --max_epochs 2 --export_dir ./export/mmjepa_bart_dinov2
```

## CLI Arguments (Selected)
- Data/Runtime
  - `--batch_size` (default 16)
  - `--num_workers` (default 4)
  - `--max_train_samples`, `--max_val_samples` (subset for quick tests)
  - `--fast_dev_run` (sanity test)
- Optimization
  - `--max_epochs` (default 3)
  - `--lr_img` (default 3e-4), `--lr_dec` (default 3e-5)
  - `--use_cosine_schedule`, `--warmup_ratio` (default 0.06)
  - `--gradient_checkpointing`, `--accumulate_grad_batches`, `--gradient_clip_val`
- Loss/Model
  - `--lambda_align` (default 0.5), `--lambda_ce` (default 1.0)
  - `--align_warmup_epochs` (default 1)
  - `--m_image_tokens` (default 8)
- I/O
  - `--log_dir`, `--ckpt_dir`, `--export_dir`

## How It Works (High-Level)
- Text teacher: `BartModel` encoder produces token states and a pooled text embedding via masked mean pooling (L2-normalized). The pooled embedding is the alignment target.
- Image student: `Dinov2Model` produces a global image embedding (CLS or mean over patches, L2-normalized).
- Alignment: an optional linear adapter maps image embedding to text dimension; cosine distance loss encourages alignment.
- Generation: an MLP maps the image embedding to M visual tokens in text dimension; tokens are concatenated before teacher encoder tokens and consumed by the BART decoder via cross-attention. Teacher forcing cross-entropy trains decoding.

## Troubleshooting
- First run is slow due to downloads; subsequent runs use cache.
- If dataloader hangs, reduce `--num_workers` (e.g., 2–4).
- If macOS MPS shows fallbacks, allow them:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## License
This repository provides research code. Datasets and pretrained models follow their respective licenses on Hugging Face.
