import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import (
    BartModel,
    BartForConditionalGeneration,
    Dinov2Model,
)
from transformers.modeling_outputs import BaseModelOutput


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


class MultimodalJEPA(pl.LightningModule):
    """
    Joint multimodal JEPA:
    - Stage-1: alignment of image embedding to text embedding from a frozen BART encoder (teacher)
    - Stage-2: generation via BART decoder using a visual prefix fused with teacher encoder token states
    """

    def __init__(
        self,
        *,
        image_model_name: str = "facebook/dinov2-base",
        text_model_name: str = "facebook/bart-base",
        m_image_tokens: int = 8,
        lambda_align: float = 0.5,
        lambda_ce: float = 1.0,
        align_warmup_epochs: int = 1,
        lr_img: float = 3e-4,
        lr_dec: float = 3e-5,
        weight_decay: float = 0.01,
        use_cosine_schedule: bool = False,
        warmup_ratio: float = 0.06,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Text teacher (frozen encoder)
        self.teacher_bart: BartModel = BartModel.from_pretrained(text_model_name)
        for p in self.teacher_bart.parameters():
            p.requires_grad = False
        self.teacher_bart.eval()

        # Generator: BART decoder (trainable). We'll provide encoder_outputs manually.
        self.generator_bart: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(
            text_model_name
        )
        # Freeze generator encoder to be safe (we don't use it). Keep shared/decoder trainable.
        for n, p in self.generator_bart.model.encoder.named_parameters():
            p.requires_grad = False
        if gradient_checkpointing:
            try:
                self.generator_bart.gradient_checkpointing_enable()
            except Exception:
                pass

        d_txt = self.generator_bart.config.d_model

        # Image encoder (trainable)
        self.image_encoder: Dinov2Model = Dinov2Model.from_pretrained(image_model_name)
        if gradient_checkpointing:
            try:
                self.image_encoder.gradient_checkpointing_enable()
            except Exception:
                pass
        d_img = self.image_encoder.config.hidden_size

        # Alignment adapter (image -> text dim)
        self.img_to_txt = nn.Linear(d_img, d_txt) if d_img != d_txt else Identity()

        # Visual prefix MLP to M image tokens in text space
        self.m_image_tokens = m_image_tokens
        self.visual_prefix = nn.Sequential(
            nn.Linear(d_img, d_txt),
            nn.GELU(),
            nn.Linear(d_txt, d_txt * m_image_tokens),
        )
        self.prefix_ln = nn.LayerNorm(d_txt)

        # Loss weights and warmup
        self.lambda_align = float(lambda_align)
        self.lambda_ce = float(lambda_ce)
        self.align_warmup_epochs = int(align_warmup_epochs)

        # Optim config
        self.lr_img = lr_img
        self.lr_dec = lr_dec
        self.weight_decay = weight_decay
        self.use_cosine_schedule = use_cosine_schedule
        self.warmup_ratio = warmup_ratio

        # Will be populated in setup if scheduler enabled
        self._total_steps: Optional[int] = None

    @torch.no_grad()
    def export_hf_pretrained(self, save_dir: str) -> None:
        """Save the fine-tuned components in HuggingFace-compatible folders.

        Layout:
          save_dir/
            image_encoder/            <- Dinov2Model.save_pretrained
            generator_bart/           <- BartForConditionalGeneration.save_pretrained
            teacher_bart/             <- optional: BartModel.save_pretrained (frozen teacher)
            adapters/
              img_to_txt.pt
              visual_prefix.pt
              prefix_ln.pt
              mm_config.json          <- minimal config for adapters
        """
        import os
        import json

        os.makedirs(save_dir, exist_ok=True)

        # Save HF models
        img_dir = os.path.join(save_dir, "image_encoder")
        gen_dir = os.path.join(save_dir, "generator_bart")
        teach_dir = os.path.join(save_dir, "teacher_bart")
        self.image_encoder.save_pretrained(img_dir)
        self.generator_bart.save_pretrained(gen_dir)
        # Teacher is optional but useful for reproducibility
        try:
            self.teacher_bart.save_pretrained(teach_dir)
        except Exception:
            pass

        # Save adapters/projections
        adapters_dir = os.path.join(save_dir, "adapters")
        os.makedirs(adapters_dir, exist_ok=True)
        # state_dicts
        if isinstance(self.img_to_txt, nn.Linear):
            torch.save(self.img_to_txt.state_dict(), os.path.join(adapters_dir, "img_to_txt.pt"))
        else:
            # Identity has no state
            torch.save({}, os.path.join(adapters_dir, "img_to_txt.pt"))
        torch.save(self.visual_prefix.state_dict(), os.path.join(adapters_dir, "visual_prefix.pt"))
        torch.save(self.prefix_ln.state_dict(), os.path.join(adapters_dir, "prefix_ln.pt"))

        # minimal config
        cfg = {
            "m_image_tokens": int(self.m_image_tokens),
            "lambda_align": float(self.lambda_align),
            "lambda_ce": float(self.lambda_ce),
            "align_warmup_epochs": int(self.align_warmup_epochs),
            "lr_img": float(self.lr_img),
            "lr_dec": float(self.lr_dec),
            "weight_decay": float(self.weight_decay),
            "use_cosine_schedule": bool(self.use_cosine_schedule),
            "warmup_ratio": float(self.warmup_ratio),
            "text_model_name": getattr(self.generator_bart.config, "name_or_path", "facebook/bart-base"),
            "image_model_name": getattr(self.image_encoder.config, "name_or_path", "facebook/dinov2-base"),
        }
        with open(os.path.join(adapters_dir, "mm_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            enc_out = self.teacher_bart.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            token_states = enc_out.last_hidden_state  # [B, L, d_txt]
            pooled = masked_mean_pool(token_states, attention_mask)
            pooled = F.normalize(pooled, dim=-1)
        return token_states, pooled

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        img_out = self.image_encoder(pixel_values=pixel_values, return_dict=True)
        # Prefer CLS token if available; otherwise mean over sequence
        token_states = img_out.last_hidden_state  # [B, T, d_img]
        if token_states.size(1) >= 1:
            img_emb = token_states[:, 0]
        else:
            img_emb = token_states.mean(dim=1)
        img_emb = F.normalize(img_emb, dim=-1)
        return img_emb

    def _make_visual_prefix_tokens(self, img_emb: torch.Tensor) -> torch.Tensor:
        B = img_emb.size(0)
        d_txt = self.generator_bart.config.d_model
        pref = self.visual_prefix(img_emb)  # [B, d_txt * M]
        pref = pref.view(B, self.m_image_tokens, d_txt)
        pref = self.prefix_ln(pref)
        return pref

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Text teacher encodes the question for token states used by the decoder cross-attention
        text_token_states, _ = self._encode_text(
            input_ids=batch["question_input_ids"], attention_mask=batch["question_attention_mask"]
        )
        # For alignment target, allow priority fallback tokens (align_*), separate from question
        _, text_emb = self._encode_text(
            input_ids=batch.get("align_input_ids", batch["question_input_ids"]),
            attention_mask=batch.get("align_attention_mask", batch["question_attention_mask"]),
        )

        # Image forward
        z_img = self._encode_image(batch["pixel_values"])  # normalized
        pred_img_txt = self.img_to_txt(z_img)
        pred_img_txt = F.normalize(pred_img_txt, dim=-1)

        # Visual prefix tokens for generation
        img_tokens = self._make_visual_prefix_tokens(z_img)

        # Augment teacher encoder token states with visual prefix
        enc_tokens = torch.cat([img_tokens, text_token_states], dim=1)
        # Build attention mask: ones for image tokens + text attention mask
        B = enc_tokens.size(0)
        device = enc_tokens.device
        img_mask = torch.ones((B, self.m_image_tokens), dtype=batch["question_attention_mask"].dtype, device=device)
        enc_attn_mask = torch.cat([img_mask, batch["question_attention_mask"]], dim=1)

        # Generator forward (decoder CE)
        labels = batch["labels"]  # already padded; pad tokens should be -100 where ignored
        decoder_attention_mask = batch.get("answer_attention_mask")

        gen_out = self.generator_bart(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_tokens),
            attention_mask=enc_attn_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        return {
            "pred_img_emb": pred_img_txt,
            "text_emb": text_emb,
            "logits": gen_out.logits,
            "ce_loss": gen_out.loss,
        }

    def _compute_alignment_loss(self, pred_img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Both inputs are normalized
        cos = F.cosine_similarity(pred_img_emb, text_emb, dim=-1)  # [B]
        loss = (1.0 - cos).mean()
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        outputs = self.forward(batch)
        l_align = self._compute_alignment_loss(outputs["pred_img_emb"], outputs["text_emb"])  # Stage-1
        l_ce = outputs["ce_loss"]  # Stage-2

        current_epoch = int(self.current_epoch)
        lambda_ce = 0.0 if current_epoch < self.align_warmup_epochs else float(self.lambda_ce)
        loss = float(self.lambda_align) * l_align + lambda_ce * l_ce

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_align", l_align, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_ce", l_ce, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        outputs = self.forward(batch)
        l_align = self._compute_alignment_loss(outputs["pred_img_emb"], outputs["text_emb"])  # Stage-1
        l_ce = outputs["ce_loss"]  # Stage-2
        lambda_ce = 0.0 if int(self.current_epoch) < self.align_warmup_epochs else float(self.lambda_ce)
        loss = float(self.lambda_align) * l_align + lambda_ce * l_ce

        # Optionally compute token-level accuracy
        with torch.no_grad():
            logits = outputs["logits"]
            labels = batch["labels"]
            # Only consider non -100 positions
            mask = labels.ne(-100)
            if mask.any():
                pred = logits.argmax(dim=-1)
                correct = (pred.eq(labels) & mask).sum().float()
                total = mask.sum().float().clamp(min=1.0)
                acc = (correct / total)
            else:
                acc = torch.tensor(0.0, device=logits.device)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_align", l_align, prog_bar=False, on_epoch=True)
        self.log("val_ce", l_ce, prog_bar=False, on_epoch=True)
        self.log("val_token_acc", acc, prog_bar=False, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):  # type: ignore[override]
        # Group params: higher LR for image encoder + adapters; lower LR for decoder
        img_modules = [self.image_encoder, self.img_to_txt, self.visual_prefix, self.prefix_ln]
        img_params = []
        for m in img_modules:
            img_params.extend(list(p for p in m.parameters() if p.requires_grad))

        dec_params = []
        for name, p in self.generator_bart.named_parameters():
            if name.startswith("model.encoder"):
                continue
            if p.requires_grad:
                dec_params.append(p)

        optim_groups = [
            {"params": img_params, "lr": self.lr_img, "weight_decay": self.weight_decay},
            {"params": dec_params, "lr": self.lr_dec, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optim_groups)

        if not self.use_cosine_schedule:
            return optimizer

        # Create cosine schedule with warmup after we know total steps; Lightning supplies this in setup.
        if self._total_steps is None:
            return optimizer

        from transformers import get_cosine_schedule_with_warmup

        warmup_steps = max(1, int(self.warmup_ratio * self._total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=self._total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_warmup",
            },
        }

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        # Determine total steps for scheduler if enabled
        if self.use_cosine_schedule and self.trainer is not None:
            # estimated_stepping_batches exists in recent Lightning
            try:
                self._total_steps = int(self.trainer.estimated_stepping_batches)
            except Exception:
                if self.trainer.max_steps is not None and self.trainer.max_steps > 0:
                    self._total_steps = int(self.trainer.max_steps)
                else:
                    self._total_steps = None


