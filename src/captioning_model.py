import pytorch_lightning as pl
import torch
from transformers import VisionEncoderDecoderModel, BartTokenizer, AutoFeatureExtractor
from .model import JEPA


class ImageCaptioningModel(pl.LightningModule):
    """
    PyTorch Lightning module for Image Captioning fine-tuning.

    This model uses a VisionEncoderDecoderModel architecture, loading the
    pre-trained image encoder weights from a JEPA checkpoint.
    """
    def __init__(
        self,
        jepa_checkpoint_path: str,
        text_decoder_name: str = "facebook/bart-base",
        learning_rate: float = 5e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load the pretrained JEPA model to extract its image encoder
        try:
            jepa_model = JEPA.load_from_checkpoint(jepa_checkpoint_path)
            image_encoder = jepa_model.image_encoder
            image_encoder_name = jepa_model.hparams.image_encoder_name
            print(f"Successfully loaded image encoder from JEPA checkpoint: {jepa_checkpoint_path}")
        except FileNotFoundError:
            print(f"Error: JEPA checkpoint not found at {jepa_checkpoint_path}")
            raise
        except Exception as e:
            print(f"An error occurred while loading the JEPA checkpoint: {e}")
            raise

        # Initialize the VisionEncoderDecoderModel, using the pre-trained image encoder
        # and a pre-trained text decoder.
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=None,  # We provide the model object directly
            encoder_model=image_encoder,
            decoder_pretrained_model_name_or_path=text_decoder_name,
        )

        # The feature extractor and tokenizer must match the models used.
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_name)
        self.tokenizer = BartTokenizer.from_pretrained(text_decoder_name)

        # Set model's generation config to use the tokenizer's special tokens
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id


    def forward(self, image_pixels, labels):
        """
        Performs a forward pass through the model.
        The model automatically computes the cross-entropy loss when `labels` are provided.
        """
        return self.model(pixel_values=image_pixels, labels=labels).loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        image_pixels = batch["image"]
        text_data = batch["text"]

        # Prepare labels for loss calculation:
        # Hugging Face models expect -100 for tokens that should be ignored (i.e., padding).
        labels = text_data["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        loss = self(image_pixels=image_pixels, labels=labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step and generates a sample caption.
        """
        image_pixels = batch["image"]
        text_data = batch["text"]
        labels = text_data["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        loss = self(image_pixels=image_pixels, labels=labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate a caption for the first image in the batch to monitor progress
        if batch_idx == 0:
            generated_ids = self.model.generate(pixel_values=image_pixels[:1], max_length=50)
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"\nSample Generated Caption: {generated_text}")

        return loss

    @torch.no_grad()
    def generate_caption(self, image_path: str):
        """
        Generates a caption for a given image file.
        """
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=64)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def configure_optimizers(self):
        """
        Configures the optimizer and an optional learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
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
