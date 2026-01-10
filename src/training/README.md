# Training Module

This module handles the training of summarization models.

## `train_matchsum.py`

Trains the **Extractive Model** (MatchSum).
-   **Architecture**: Uses a Siamese BERT structure to score candidate summaries against the document.
-   **Method**: Margin Ranking Loss. The model learns to assign a higher score to the "gold" summary than to a random negative summary.
-   **Input**: Triads of (Document, Positive Summary, Negative Summary).

## `train_abstractive.py`

Trains the **Abstractive Model** (mT5/PEGASUS).
-   **Architecture**: Seq2Seq Transformer (`google/mt5-small`).
-   **Method**: Supervised Fine-Tuning.
-   **Input**: Document -> Summary.
-   **Optimization**: Uses Gradient Accumulation to handle batch sizes on smaller GPUs.

## Usage
Training can be triggered via `main.py` if the corresponding flags are enabled and imports uncommented.
```bash
python main.py --train-ext
python main.py --train-abs
```
