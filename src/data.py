import random
from typing import List, Dict, Any

import PIL
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer, BatchFeature, BatchEncoding


class CocoDataset(Dataset):
    """
    A PyTorch Dataset for the COCO Captions dataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "image": sample["image"],
            "text": sample["sentences_raw"]
        }


class CocoDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the MS COCO Captions dataset.
    """

    def __init__(
            self,
            batch_size: int = 16,
            image_processor_name: str = "facebook/dinov2-base",
            text_processor_name: str = "facebook/bart-base",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
        self.text_processor = AutoTokenizer.from_pretrained(text_processor_name)

    def prepare_data(self):
        """
        Downloads the dataset.
        """
        load_dataset("Multimodal-Fatima/COCO_captions_test", split="test")

    def setup(self, stage: str = None):
        """
        Assigns the dataset to the respective stage.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = CocoDataset(load_dataset("Multimodal-Fatima/COCO_captions_train", split="train"))
            self.val_dataset = CocoDataset(load_dataset("Multimodal-Fatima/COCO_captions_validation", split="validation"))
        if stage == "test" or stage is None:
            self.test_dataset = CocoDataset(load_dataset("Multimodal-Fatima/COCO_captions_test", split="test"))

    def process_images(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        Processes a list of images using the image processor.
        """
        processed: BatchFeature = self.image_processor(
            images=images,
            size={"height": 224, "width": 224},
            do_resize=True,
            do_center_crop=False,
            do_rescale=True,
            do_normalize=True,
            return_tensors="pt"
        )
        return processed.pixel_values

    def process_text(self, sentences_raw: List[str]) -> BatchEncoding:
        """
        Processes a list of sentences using the text tokenizer.
        """
        processed: BatchEncoding = self.text_processor(
            sentences_raw,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return processed

    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of samples into a dictionary of tensors.
        """
        images = [sample["image"] for sample in samples]
        texts = [sample["text"] for sample in samples]
        flattened_texts = [random.choice(t) if isinstance(t, list) else t for t in texts]

        return {
            "image": self.process_images(images),
            "text": self.process_text(flattened_texts)
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
