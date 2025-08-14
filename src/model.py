import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel

from .predictor import Predictor


def random_sample_indices(b: int, t: int, prob: float, device=None) -> torch.Tensor:
    """
    Get random token indices per batch.
    """
    n = max(1, int(round(prob * t)))
    idx = torch.rand(b, t, device=device).argsort(dim=1)[:, :n]
    return idx


class JEPA(pl.LightningModule):
    """
    A PyTorch Lightning module for the Multimodal Joint-Embedding Predictive Architecture (JEPA).

    This model integrates a student vision encoder, a frozen teacher text encoder, and a predictor
    to learn a joint embedding space for images and text.
    """

    def __init__(
            self,
            image_encoder_name: str = "facebook/dinov2-base",
            text_encoder_name: str = "facebook/bart-base",
            predictor_embed_dim: int = 512,
            predictor_num_heads: int = 8,
            predictor_depth: int = 6,
            context_prob: float = 0.7,
            target_prob: float = 0.9,
            learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Image Encoder (Student)
        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)

        # Text Encoder (Teacher)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name).encoder
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
        context_idxs = random_sample_indices(b=b, t=t, prob=self.hparams.context_prob, device=self.device)
        context_embeddings = image_embeddings[torch.arange(b)[:, None], context_idxs]

        # Get targets and masks from text
        b, t, d = text_embeddings.shape
        mask_token_expanded = self.mask_token.expand(b, t, d)
        positions = self.text_encoder.embed_positions(mask_token_expanded)
        masks_with_positions = mask_token_expanded + positions

        target_idxs = random_sample_indices(b=b, t=t, prob=self.hparams.target_prob, device=self.device)
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
        Configures the optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
