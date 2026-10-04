"""LLaVA-format conversation datasets for discrete MLLM training."""

from __future__ import annotations

import os

import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, Dataset

from vision_encoder_eval.mllm.discrete.data.qwen_chat import IGNORE_INDEX, IMAGE_TOKEN_INDEX, preprocess_qwen_conversation
from vision_encoder_eval.mllm.utils.data_mix import apply_sampling_strategy, load_llava_json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LLaVADataset(Dataset):
    """LLaVA conversation dataset backed by a single JSON source."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_size: int = 256,
        max_length: int = 2048,
        *,
        samples: list[dict] | None = None,
        sampling_strategy: str = "all",
        sampling_seed: int = 42,
    ):
        if samples is None:
            raw = load_llava_json(data_path)
            raw = apply_sampling_strategy(raw, sampling_strategy, seed=sampling_seed)
            samples = [item for item in raw if item.get("image")]
            skipped = len(raw) - len(samples)
            if skipped:
                print(
                    f"Filtered {skipped} text-only samples from {data_path}, "
                    f"keeping {len(samples)}",
                    flush=True,
                )
        self.datas = samples
        self.data_path = data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        self._bad_image_warned: set[str] = set()

    def _load_image(self, img_path: str) -> torch.Tensor | None:
        try:
            with Image.open(img_path) as img:
                return self.transform(img.convert("RGB"))
        except (OSError, FileNotFoundError, ValueError) as exc:
            if img_path not in self._bad_image_warned:
                self._bad_image_warned.add(img_path)
                print(f"Skipping unreadable image {img_path}: {exc}", flush=True)
            return None

    def __len__(self) -> int:
        return len(self.datas)

    def _tokenize_conversation(self, item: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return preprocess_qwen_conversation(
            item["conversations"],
            self.tokenizer,
            max_length=self.max_length,
        )

    def __getitem__(self, idx: int) -> dict:
        n = len(self.datas)
        for offset in range(n):
            item = self.datas[(idx + offset) % n]
            img_path = os.path.join(self.image_folder, item["image"])
            pixel_values = self._load_image(img_path)
            if pixel_values is None:
                continue
            input_ids, labels = self._tokenize_conversation(item)
            return {"pixel_values": pixel_values, "input_ids": input_ids, "labels": labels}
        raise RuntimeError(f"No readable images found in dataset {self.data_path}")


def build_training_dataset(
    *,
    datasets: list[dict],
    tokenizer,
    image_size: int,
    max_length: int,
    seed: int = 42,
) -> Dataset:
    """Build one or more datasets; multiple sources are concatenated and shuffled each epoch."""
    if not datasets:
        raise ValueError("At least one dataset entry is required")

    parts: list[Dataset] = []
    for entry in datasets:
        part = LLaVADataset(
            data_path=entry["data_path"],
            image_folder=entry["image_folder"],
            tokenizer=tokenizer,
            image_size=image_size,
            max_length=max_length,
            sampling_strategy=entry.get("sampling_strategy", "all"),
            sampling_seed=seed,
        )
        if len(part) == 0:
            raise ValueError(f"Dataset has no usable samples: {entry['data_path']}")
        print(
            f"Loaded {len(part)} samples from {entry['data_path']} "
            f"(images: {entry['image_folder']})",
            flush=True,
        )
        parts.append(part)

    if len(parts) == 1:
        return parts[0]
    total = sum(len(part) for part in parts)
    print(
        f"Mixed finetune dataset: {len(parts)} sources, {total} samples total "
        f"(random interleaving via DataLoader shuffle)",
        flush=True,
    )
    return ConcatDataset(parts)


def collate_fn(batch: list[dict]) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX
    )
    attention_mask = input_ids.ne(0)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
