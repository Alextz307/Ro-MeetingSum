import logging
import torch
import gc
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from src.utils.data_loader import RoAbstractiveDataset
from src.config import CLEAN_FILE, MODELS_DIR


# --- CONFIG ---
MODEL_NAME = "google/mt5-small"
OUTPUT_DIR = str(MODELS_DIR / "checkpoints_abs")
EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LR = 5e-4


def train_abstractive() -> None:
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    logging.info("Starting Abstractive Training")

    dataset = RoAbstractiveDataset(
        file_path=CLEAN_FILE,
        model_name=MODEL_NAME,
        max_input_len=512,
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        predict_with_generate=False,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=EPOCHS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        fp16=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=dataset.tokenizer, model=model)  # type: ignore

    trainer = Seq2SeqTrainer(
        model=model,  # type: ignore
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=dataset.tokenizer,  # type: ignore
        data_collator=data_collator,
    )

    trainer.train()

    final_path = str(MODELS_DIR / "mt5_finetuned")
    model.save_pretrained(final_path)
    dataset.tokenizer.save_pretrained(final_path)
    logging.info(f"Abstractive Model Saved to {final_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_abstractive()
