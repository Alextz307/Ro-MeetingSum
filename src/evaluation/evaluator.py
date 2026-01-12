import logging
import json
import torch
from transformers import pipeline, AutoTokenizer, PreTrainedTokenizer
from rouge_score import rouge_scorer  # type: ignore

from src.config import CLEAN_FILE, EXTRACTIVE_MODEL_PATH, ABSTRACTIVE_MODEL_PATH
from src.inference.extract_summary import ExtractiveSummarizer
from src.processing.post_processing import clean_text_for_generation
from src.inference.chunking import summarize_in_chunks


class ModelEvaluator:
    """
    Orchestrates the evaluation of different summarization strategies against ground truth.
    Compares:
    1. MatchSum (Extractive)
    2. Hybrid (Extractive + Abstractive)
    3. Baseline (Blind Slicing)
    """

    def __init__(self, test_size: int):
        """
        Args:
            test_size (int): Number of items from the dataset to evaluate.
        """

        self.test_size = test_size
        self.device = (
            0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
        )

    def run(self) -> None:
        """
        Runs the full evaluation loop, prints individual results, and displays a final score table.
        """

        logging.info("\n --- FINAL COMPARISON: Hybrid vs. Blind Baseline ---")

        ext_summarizer = ExtractiveSummarizer(str(EXTRACTIVE_MODEL_PATH))

        try:
            abs_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                str(ABSTRACTIVE_MODEL_PATH), fix_mistral_regex=True
            )
        except Exception:
            abs_tokenizer = AutoTokenizer.from_pretrained(str(ABSTRACTIVE_MODEL_PATH))

        abs_pipeline = pipeline(
            "summarization",
            model=str(ABSTRACTIVE_MODEL_PATH),
            tokenizer=abs_tokenizer,  # type: ignore
            device=self.device,
        )

        try:
            with open(CLEAN_FILE, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            logging.error("Clean file not found.")
            return

        test_data = data[: self.test_size]

        print(f"\nEvaluating on {len(test_data)} meetings...")

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        ext_scores: dict[str, list[float]] = {"r1": [], "rl": []}
        hybrid_scores: dict[str, list[float]] = {"r1": [], "rl": []}
        baseline_scores: dict[str, list[float]] = {"r1": [], "rl": []}

        for i, entry in enumerate(test_data):
            source_text: str = entry["abs_source"]
            ground_truth: str = entry["abs_target"]

            print(
                f"\n[{i+1}/{self.test_size}] Processing meeting ({len(source_text)} chars)..."
            )

            summary_ext: str = ""

            # --- A. Extractive (MatchSum) ---
            try:
                summary_ext = ext_summarizer.summarize(
                    source_text, ratio=0.15, min_k=20, max_k=80
                )
                s_ext = scorer.score(ground_truth, summary_ext)
                ext_scores["r1"].append(float(s_ext["rouge1"].fmeasure))  # type: ignore
                ext_scores["rl"].append(float(s_ext["rougeL"].fmeasure))  # type: ignore
            except Exception as e:
                print(f"  Extractive Failed: {e}")
                continue

            # --- B. Hybrid (MatchSum + Filter + Chunking) ---
            if summary_ext:
                try:
                    clean_input = clean_text_for_generation(summary_ext)
                    summary_hybrid = summarize_in_chunks(
                        abs_pipeline, clean_input, abs_tokenizer
                    )

                    s_hybrid = scorer.score(ground_truth, summary_hybrid)
                    hybrid_scores["r1"].append(float(s_hybrid["rouge1"].fmeasure))  # type: ignore
                    hybrid_scores["rl"].append(float(s_hybrid["rougeL"].fmeasure))  # type: ignore

                    if i == 0:
                        print("\n" + "=" * 20 + " EXAMPLE HYBRID SUMMARY " + "=" * 20)
                        print(summary_hybrid)
                        print("=" * 64 + "\n")
                except Exception as e:
                    print(f"  Hybrid Failed: {e}")

            # --- C. Baseline (Raw Blind Slicing + Chunking) ---
            try:
                BUDGET = 15000
                L = len(source_text)

                if L <= BUDGET:
                    raw_blind_input = source_text
                else:
                    slice_size = BUDGET // 3
                    start_part = source_text[:slice_size]
                    mid_start = (L // 2) - (slice_size // 2)
                    mid_end = mid_start + slice_size
                    mid_part = source_text[mid_start:mid_end]
                    end_part = source_text[-slice_size:]

                    raw_blind_input = start_part + " " + mid_part + " " + end_part

                summary_baseline = summarize_in_chunks(
                    abs_pipeline, raw_blind_input, abs_tokenizer
                )

                s_base = scorer.score(ground_truth, summary_baseline)
                baseline_scores["r1"].append(float(s_base["rouge1"].fmeasure))  # type: ignore
                baseline_scores["rl"].append(float(s_base["rougeL"].fmeasure))  # type: ignore
            except Exception as e:
                print(f"  Baseline Failed: {e}")

            if i == 0:
                print("\n" + "=" * 20 + " EXAMPLE SUMMARIES (ITEM 0) " + "=" * 20)
                print(f"\n[1] EXTRACTIVE (MatchSum):\n{summary_ext}\n")
                print("-" * 60)
                if "summary_hybrid" in locals():
                    print(f"\n[2] HYBRID (MatchSum + mT5):\n{summary_hybrid}\n")
                else:
                    print("\n[2] HYBRID: (Skipped/Failed)\n")
                print("-" * 60)
                if "summary_baseline" in locals():
                    print(f"\n[3] BASELINE (Blind Check):\n{summary_baseline}\n")
                else:
                    print("\n[3] BASELINE: (Skipped/Failed)\n")
                print("=" * 64 + "\n")

        def get_avg(scores: list[float]) -> float:
            return sum(scores) / len(scores) if len(scores) > 0 else 0.0

        print("\n" + "=" * 60)
        print(f"FINAL RESULTS TABLE (N={len(test_data)})")
        print("=" * 60)
        print(f"{'METHOD':<30} | {'ROUGE-1':<10} | {'ROUGE-L':<10}")
        print("-" * 60)
        print(
            f"{'1. Baseline (Blind Slicing)':<30} | {get_avg(baseline_scores['r1']):.4f}     | {get_avg(baseline_scores['rl']):.4f}"
        )
        print(
            f"{'2. Extractive (MatchSum)':<30} | {get_avg(ext_scores['r1']):.4f}     | {get_avg(ext_scores['rl']):.4f}"
        )
        print(
            f"{'3. Hybrid (Intelligent Filter)':<30} | {get_avg(hybrid_scores['r1']):.4f}     | {get_avg(hybrid_scores['rl']):.4f}"
        )
        print("=" * 60)
        print(f"FINAL RESULTS TABLE (N={len(test_data)})")
        print("=" * 60)
        print(f"{'METHOD':<30} | {'ROUGE-1':<10} | {'ROUGE-L':<10}")
        print("-" * 60)
        print(
            f"{'1. Baseline (Blind Slicing)':<30} | {get_avg(baseline_scores['r1']):.4f}     | {get_avg(baseline_scores['rl']):.4f}"
        )
        print(
            f"{'2. Extractive (MatchSum)':<30} | {get_avg(ext_scores['r1']):.4f}     | {get_avg(ext_scores['rl']):.4f}"
        )
        print(
            f"{'3. Hybrid (Intelligent Filter)':<30} | {get_avg(hybrid_scores['r1']):.4f}     | {get_avg(hybrid_scores['rl']):.4f}"
        )
        print("=" * 60)
