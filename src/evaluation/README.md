# Evaluation Module

This module handles the comparative evaluation of the summarization pipelines.

## `evaluator.py`

The `ModelEvaluator` class runs the evaluation loop:
1.  **Loading Models**: Loads the Extractive (MatchSum) and Abstractive (mT5/Pegasus) models.
2.  **Dataset**: Loads the cleaned test set.
3.  **Metrics**: Uses ROUGE (1 and L) to score summaries against ground truth.
4.  **Pipelines Evaluated**:
    -   **Baseline (Blind Slicing)**: Takes first, middle, and last parts of the transcript and summarizes them.
    -   **Extractive Only**: Uses MatchSum to extract sentences.
    -   **Hybrid**: Uses MatchSum to extract sentences -> Filters/Clean -> Abstractive Summarization.

### Usage
Run normalization via CLI:
```bash
python main.py --evaluate --test-size 64
```
