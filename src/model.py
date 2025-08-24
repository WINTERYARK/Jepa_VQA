import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel

from .predictor import Predictor
from typing import Tuple, Union
import random


def random_sample_indices(b: int, t: int, prob: float, device=None) -> torch.Tensor:
    """
    Get random token indices per batch.
    """
    n = max(1, int(round(prob * t)))
    idx = torch.rand(b, t, device=device).argsort(dim=1)[:, :n]
    return idx


class JEPA(pl.LightningModule):

    def __init__(
            self,
            image_encoder_name: str = "facebook/dinov2-base",
            text_encoder_name: str = "facebook/bart-base",
            predictor_embed_dim: int = 512,
            predictor_num_heads: int = 8,
            predictor_depth: int = 6,
            context_prob: Union[float, Tuple[float, float]] = (0.7, 1.0), # 0.7-1.0
            target_prob: Union[float, Tuple[float, float]] = (0.7, 1.0),
            learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Image Encoder (Student)
        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)

        # Text Encoder (Teacher)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name).encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False  # Freeze the text encoder

        self.embed_dim = self.image_encoder.config.hidden_size

        # Predictor
        self.predictor = Predictor(
            embed_dim=self.embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            num_heads=predictor_num_heads,
            depth=predictor_depth,
        )

        # Mask token for text embeddings
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, image, text):
        """
        Performs a forward pass through the model.
        """
        # Get image embeddings (context)
        image_embeddings = self.image_encoder(image).last_hidden_state

        # Get text embeddings (targets)
        text_embeddings = self.text_encoder(**text).last_hidden_state

        # Get context from images
        b, t, _ = image_embeddings.shape
        
        # Determine sampling probability, randomizing if a range is given
        if isinstance(self.hparams.context_prob, tuple):
            min_prob, max_prob = self.hparams.context_prob
            context_prob = random.uniform(min_prob, max_prob)
        else:
            context_prob = self.hparams.context_prob
            
        context_idxs = random_sample_indices(b=b, t=t, prob=context_prob, device=self.device)
        context_embeddings = image_embeddings[torch.arange(b)[:, None], context_idxs]

        # Get targets and masks from text
        b, t, d = text_embeddings.shape
        mask_token_expanded = self.mask_token.expand(b, t, d)
        positions = self.text_encoder.embed_positions(mask_token_expanded)
        masks_with_positions = mask_token_expanded + positions

        # Determine sampling probability, randomizing if a range is given
        if isinstance(self.hparams.target_prob, tuple):
            min_prob, max_prob = self.hparams.target_prob
            target_prob = random.uniform(min_prob, max_prob)
        else:
            target_prob = self.hparams.target_prob

        target_idxs = random_sample_indices(b=b, t=t, prob=target_prob, device=self.device)
        targets = text_embeddings[torch.arange(b)[:, None], target_idxs]
        target_masks = masks_with_positions[torch.arange(b)[:, None], target_idxs]

        # Predict text embeddings from image context
        predictions = self.predictor(context_embeddings, target_masks)

        return predictions, targets

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        predictions, targets = self.forward(batch["image"], batch["text"])
        loss = nn.functional.mse_loss(predictions, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        """
        predictions, targets = self.forward(batch["image"], batch["text"])
        loss = nn.functional.mse_loss(predictions, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

