import json
import torch
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from typing import Any, cast
from pathlib import Path


class RoMatchSumDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Advanced Dataset for Siamese Training.
    Returns a triad: (Document, Positive Summary, Negative Summary).
    """

    def __init__(
        self,
        file_path: str | Path,
        model_name: str = "dumitrescustefan/bert-base-romanian-cased-v1",
        max_doc_len: int = 512,
        max_sum_len: int = 128,
        mode: str = "train",  # 'train' or 'inference'
    ):
        self.file_path = Path(file_path)
        self.max_doc_len = max_doc_len
        self.max_sum_len = max_sum_len
        self.mode = mode

        print(f"📉 Loading Tokenizer: {model_name}...")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.data: list[dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def _tokenize(self, text: str, max_len: int) -> dict[str, torch.Tensor]:
        """Helper to strictly tokenize text into Tensors."""
        encoding: BatchEncoding = self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Explicit casting for strict typing
        return {
            "input_ids": cast(torch.Tensor, encoding["input_ids"]).squeeze(0),
            "attention_mask": cast(torch.Tensor, encoding["attention_mask"]).squeeze(0),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # 1. Prepare Document (Source)
        source_sents: list[str] = item.get("ext_source_sents", [])
        if not source_sents:
            source_sents = ["Text lipsă."]

        doc_text = " ".join(source_sents)
        doc_tensors = self._tokenize(doc_text, self.max_doc_len)

        result = {
            "doc_input_ids": doc_tensors["input_ids"],
            "doc_attention_mask": doc_tensors["attention_mask"],
        }

        # 2. If Training, Prepare Positive & Negative Summaries
        if self.mode == "train":
            # A. Positive (The Ground Truth)
            pos_text: str = item.get("ext_target_text", "")

            # B. Negative (Random sentences from the doc)
            # We pick 3 random sentences to fake a "bad summary"
            if len(source_sents) > 3:
                neg_sents = random.sample(source_sents, 3)
            else:
                neg_sents = source_sents  # Fallback
            neg_text = " ".join(neg_sents)

            pos_tensors = self._tokenize(pos_text, self.max_sum_len)
            neg_tensors = self._tokenize(neg_text, self.max_sum_len)

            result.update(
                {
                    "pos_input_ids": pos_tensors["input_ids"],
                    "pos_attention_mask": pos_tensors["attention_mask"],
                    "neg_input_ids": neg_tensors["input_ids"],
                    "neg_attention_mask": neg_tensors["attention_mask"],
                }
            )

        return result


class RoAbstractiveDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Dataset for Abstractive Summarization (mT5).
    Returns (Input_IDs, Attention_Mask, Labels).
    """

    def __init__(
        self,
        file_path: str | Path,
        model_name: str = "google/mt5-small",
        max_input_len: int = 1024,
        max_target_len: int = 128,
    ):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            self.data = raw_data

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.data)  # type: ignore

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # 1. Prepare Inputs
        # T5 expects a prefix for the task
        source_text = "summarize: " + item.get("abs_source", "")
        target_text = item.get("abs_target", "")

        # 2. Tokenize Source
        model_inputs: BatchEncoding = self.tokenizer(
            source_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 3. Tokenize Target (Labels)
        # We use text_target=... for the summary
        labels: BatchEncoding = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = cast(torch.Tensor, model_inputs["input_ids"]).squeeze(0)
        attention_mask = cast(torch.Tensor, model_inputs["attention_mask"]).squeeze(0)
        labels_ids = cast(torch.Tensor, labels["input_ids"]).squeeze(0)

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
        }
