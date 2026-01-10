# Team Contributions (Work Split)

This document outlines the individual contributions of each team member to the **Ro-MeetingSum** project. The workload was distributed evenly to ensure both members engaged with core NLP Engineering, Model Training, and Software Architecture tasks.

---

## 👨‍💻 Alexandru Lorintz
**Focus:** Abstractive Modeling, Inference Strategies, and Pipeline Integration.

1.  **Abstractive Summarization (mT5)**:
    *   Set up the fine-tuning pipeline for `google/mt5-small` using the Hugging Face `Seq2SeqTrainer`.
    *   Handled the specific tokenization requirements for T5 (e.g., `<extra_id_0>` handling) and prompt engineering ("summarize: ").
    *   Managed model checkpointing and saving logic.

2.  **Advanced Inference Strategies**:
    *   Designed the "Divide and Conquer" **Chunking Strategy** to solve the 512-token limit of Transformer models.
    *   Implemented the `summarize_in_chunks` logic with sliding windows to ensure context continuity across segment boundaries.
    *   Added **Regex Post-Processing** patterns to clean parliamentary boilerplate (titles, procedural fluff) from the final output.

3.  **System Architecture & Integration**:
    *   Refactored the codebase into a modular design (`src/` package structure).
    *   Built the central CLI (`main.py`) to orchestrate the full `Extractive -> Abstractive` data flow.
    *   Ensured type safety and robust error handling across the application.

---

## 👨‍💻 Adrian Iurian
**Focus:** Data Engineering, Extractive Modeling, and Evaluation Metrics.

1.  **Data Acquisition & Inspection**:
    *   Designed and implemented the `CDEPScraper` class to bypass legacy SSL security (`SECLEVEL=1`) on the government server.
    *   Reverse-engineered the HTML structure of `cdep.ro` to extract metadata, full transcripts, and summary ground truths.
    *   Performed initial data analysis to determine sentence length distributions.

2.  **Extractive Summarization (MatchSum)**:
    *   Implemented the Siamese BERT architecture (`RomanianMatchSum` class) using `bert-base-romanian-cased`.
    *   Developed the "Global Candidate Scoring" logic to rate sentences against the document embedding without truncation.
    *   Tuned the selection heuristics (Top-k selection, Min/Max clamping) to optimize the Context-to-Noise ratio key for the hybrid stage.

3.  **Evaluation Framework**:
    *   Implemented the `ModelEvaluator` class and integrated the **ROUGE** library.
    *   Designed the comparative experiment (Baseline vs. Extractive vs. Hybrid) to quantify performance gains.
