# Data Processing Module

This module handles the cleaning and preparation of raw parliamentary data.

## `cleaner.py`

The `DataProcessor` class is responsible for:
1.  **Loading Spacy**: Uses `ro_core_news_sm` for Romanian sentence segmentation. disabling NER and parser for speed.
2.  **Cleaning Summaries**:
    -   Merges summary points into a single text block.
    -   Removes list numbering (e.g., "1.1", "2.") and trailing semicolons.
3.  **Cleaning Transcripts**:
    -   Removes procedural noise like "(Aplauze)", "(Vociferări)".
    -   Normalizes whitespace.
4.  **Sentence Segmentation**:
    -   Splits text into sentences using Spacy `sentencizer`.
    -   Filters out very short sentences (< 5 chars).
    -   Limits input text to 100k chars to prevent memory issues.
5.  **Data Structuring**:
    -   Creates `abs_source` and `abs_target` for abstractive models.
    -   Creates `ext_source_sents` for extractive models (list of sentences).
    -   Saves the processed data to JSON.

## usage
This can be run via the main CLI or imported.
