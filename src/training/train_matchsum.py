import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm  # type: ignore
import os
import logging

from src.models.matchsum_ro import RomanianMatchSum
from src.utils.data_loader import RoMatchSumDataset
from src.config import CLEAN_FILE, CHECKPOINTS_DIR


# --- CONFIG ---
BATCH_SIZE = 4
EPOCHS = 5
LR = 2e-5
MARGIN = 0.5
SAVE_PATH = str(CHECKPOINTS_DIR)

# Auto-detect device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)


def train() -> None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    logging.info(f"Training on Device: {device}")

    logging.info("Loading Dataset...")
    dataset = RoMatchSumDataset(file_path=CLEAN_FILE, mode="train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RomanianMatchSum()
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)  # type: ignore
    loss_fct = nn.MarginRankingLoss(margin=MARGIN)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            optimizer.zero_grad()

            doc_ids = batch["doc_input_ids"].to(device)
            doc_mask = batch["doc_attention_mask"].to(device)

            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)

            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            doc_emb = model(doc_ids, doc_mask)
            pos_emb = model(pos_ids, pos_mask)
            neg_emb = model(neg_ids, neg_mask)

            score_pos = torch.nn.functional.cosine_similarity(doc_emb, pos_emb)  # type: ignore
            score_neg = torch.nn.functional.cosine_similarity(doc_emb, neg_emb)  # type: ignore

            target = torch.ones(doc_ids.size(0)).to(device)
            loss = loss_fct(score_pos, score_neg, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        save_file = f"{SAVE_PATH}/matchsum_ro_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_file)  # type: ignore
        logging.info(
            f"Epoch {epoch+1} Complete. Avg Loss: {total_loss / len(dataloader):.4f}"
        )
        logging.info(f"Model saved to {save_file}")


if __name__ == "__main__":
    train()
