import argparse
import os
import random
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data_mmjepa import create_dataloaders
from mm_jepa import MultimodalJEPA


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multimodal JEPA (joint alignment + generation)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--lr_img", type=float, default=3e-4)
    parser.add_argument("--lr_dec", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--export_dir", type=str, default=None, help="Optional directory to save HF-style pretrained artifacts")
    parser.add_argument("--precision", type=str, default=None, help="32, 16-mixed, bf16-mixed")
    parser.add_argument("--m_image_tokens", type=int, default=8)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--align_warmup_epochs", type=int, default=1)
    parser.add_argument("--use_cosine_schedule", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader, image_processor, tokenizer = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    model = MultimodalJEPA(
        m_image_tokens=args.m_image_tokens,
        lambda_align=args.lambda_align,
        lambda_ce=args.lambda_ce,
        align_warmup_epochs=args.align_warmup_epochs,
        lr_img=args.lr_img,
        lr_dec=args.lr_dec,
        weight_decay=args.weight_decay,
        use_cosine_schedule=args.use_cosine_schedule,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Precision handling
    if args.precision is None:
        precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 32
    else:
        precision = args.precision

    logger = TensorBoardLogger(save_dir=args.log_dir, name="mmjepa_joint")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="mmjepa-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_cb],
        max_epochs=args.max_epochs,
        precision=precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=True,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
    )

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if checkpoint_cb.best_model_path:
        print(f"Best checkpoint saved at: {checkpoint_cb.best_model_path}")
    if checkpoint_cb.last_model_path:
        print(f"Last checkpoint saved at: {checkpoint_cb.last_model_path}")

    # Export pretrained artifacts if requested
    if args.export_dir is not None:
        os.makedirs(args.export_dir, exist_ok=True)
        print(f"Exporting pretrained model to: {args.export_dir}")
        # Save model submodules and adapters
        model.export_hf_pretrained(args.export_dir)
        # Save processors/tokenizers alongside for inference readiness
        proc_dir = os.path.join(args.export_dir, "processors")
        os.makedirs(proc_dir, exist_ok=True)
        image_processor.save_pretrained(proc_dir)
        tokenizer.save_pretrained(proc_dir)
        print(f"Export complete at: {args.export_dir}")


if __name__ == "__main__":
    main()


