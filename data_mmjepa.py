from typing import Any, Dict, List, Optional, Tuple

import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

from transformers import AutoImageProcessor, AutoTokenizer


ALIGN_TEXT_FIELDS_PRIORITY = [
    "question",
    "caption",
    "answer",
    "text",
]


def _select_align_text(example: Dict[str, Any]) -> Optional[str]:
    for key in ALIGN_TEXT_FIELDS_PRIORITY:
        val = example.get(key)
        if isinstance(val, str) and len(val.strip()) > 0:
            return val.strip()
        # PathVQA sometimes uses dicts or lists; attempt to coerce
        if val is None:
            continue
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str):
            return val[0].strip()
        if isinstance(val, dict):
            # try common fields
            for k in ["text", "caption", "question", "answer"]:
                if k in val and isinstance(val[k], str) and len(val[k].strip()) > 0:
                    return val[k].strip()
    return None


class PathVQADataset(Dataset):
    def __init__(
        self,
        split: str,
        image_processor: AutoImageProcessor,
        tokenizer: AutoTokenizer,
        max_len_question: int = 64,
        max_len_answer: int = 16,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.split = split
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len_question = max_len_question
        self.max_len_answer = max_len_answer
        self.rng = random.Random(seed)

        raw = load_dataset("flaviagiammarino/path-vqa")
        if isinstance(raw, DatasetDict):
            if split in raw:
                ds = raw[split]
            else:
                # Build custom split 95/5
                train_size = int(0.95 * len(raw["train"]))
                ds = raw["train"].train_test_split(test_size=len(raw["train"]) - train_size, seed=seed)["test" if split == "validation" else "train"]
        else:
            ds = raw

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.data = ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.data[idx]
        # Obtain text fields
        question = ex.get("question")
        if isinstance(question, list):
            question = question[0] if question else ""
        if question is None:
            question = ""
        question = str(question)

        answer = ex.get("answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if answer is None:
            answer = ""
        answer = str(answer)

        align_text = _select_align_text(ex) or question or answer
        if align_text is None or len(align_text.strip()) == 0:
            align_text = question

        # Image
        image = ex.get("image")
        pixel_values = None
        try:
            if image is not None:
                pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        except Exception:
            pixel_values = None

        if pixel_values is None:
            # create a dummy image if corrupted/missing to keep batch intact
            # Try crop_size first, then size -> height/width fallbacks
            height = 224
            width = 224
            try:
                crop_size = getattr(self.image_processor, "crop_size", None)
                if isinstance(crop_size, dict):
                    height = int(crop_size.get("height", height))
                    width = int(crop_size.get("width", width))
                else:
                    size = getattr(self.image_processor, "size", None)
                    if isinstance(size, dict):
                        height = int(size.get("height", height))
                        width = int(size.get("width", width))
            except Exception:
                pass
            pixel_values = torch.zeros((3, height, width))

        # Tokenize question for encoder teacher
        q_tok = self.tokenizer(
            question,
            max_length=self.max_len_question,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize align_text (use the same as question for alignment target embedding)
        align_tok = self.tokenizer(
            align_text,
            max_length=self.max_len_question,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize answer for decoder labels
        # We will shift inside HF model; set pad_token_id-> -100 for CE ignore
        a_tok = self.tokenizer(
            answer,
            max_length=self.max_len_answer,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = a_tok.input_ids[0]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "question_input_ids": q_tok.input_ids[0],
            "question_attention_mask": q_tok.attention_mask[0],
            # For alignment we use align_tok tokens (could differ from question)
            "align_input_ids": align_tok.input_ids[0],
            "align_attention_mask": align_tok.attention_mask[0],
            "labels": labels,
            "answer_attention_mask": a_tok.attention_mask[0],
        }


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            collated[k] = torch.stack([x[k] for x in batch], dim=0)
        else:
            # Should not occur, but safeguard
            collated[k] = torch.tensor(batch[0][k])
    return collated


def create_dataloaders(
    batch_size: int,
    num_workers: int,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    seed: int = 42,
    max_len_question: int = 64,
    max_len_answer: int = 16,
) -> Tuple[DataLoader, DataLoader, AutoImageProcessor, AutoTokenizer]:
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    train_ds = PathVQADataset(
        split="train",
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_len_question=max_len_question,
        max_len_answer=max_len_answer,
        max_samples=max_train_samples,
        seed=seed,
    )
    val_ds = PathVQADataset(
        split="validation",
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_len_question=max_len_question,
        max_len_answer=max_len_answer,
        max_samples=max_val_samples,
        seed=seed,
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0 and pin_memory

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate_fn,
    )

    return train_loader, val_loader, image_processor, tokenizer


