# Inference Module

This module handles the inference logic for both extractive and abstractive summarization.

## `extract_summary.py`

The `ExtractiveSummarizer` class uses a loaded `RomanianMatchSum` model to extract key sentences from a text.
The workflow is:
1.  **Tokenization**: Uses `bert-base-romanian-cased-v1` tokenizer.
2.  **Sentence Splitting**: Uses Spacy to split text into sentences.
3.  **Dynamic K Selection**: Decides how many sentences to keep based on the input length and a ratio (default 20%).
4.  **Scoring**:
    -   Computes a document embedding using the first 3000 chars.
    -   Computes embeddings for each candidate sentence.
    -   Calculates Cosine Similarity between the document and each sentence.
5.  **Selection**: Picks the top-k highest scoring sentences and reorders them to match the original flow.

## `chunking.py`

Handles large texts for abstractive models by splitting them into chunks.
-   Uses a sliding window approach (stride of 450 tokens).
-   Summarizes each chunk independently using the provided pipeline.
-   Joins the chunk summaries.
