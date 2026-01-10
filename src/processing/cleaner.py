import json
import re
import logging
from pathlib import Path
import spacy
from tqdm import tqdm  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DataProcessor:
    """
    Handles the cleaning and preprocessing of raw parliamentary data.

    This class prepares data for:
    1. Abstractive Summarization: Creating clean paragraphs for Seq2Seq models.
    2. Extractive Summarization: Segmenting text into sentences for MatchSum.
    """

    def __init__(self) -> None:
        """Initializes the Spacy NLP model for Romanian."""

        print("Loading Spacy Romanian model...")
        try:
            self.nlp = spacy.load("ro_core_news_sm")
            self.nlp.disable_pipes(["ner", "parser"])
            self.nlp.add_pipe("sentencizer")
        except OSError:
            raise ImportError("Please run: python -m spacy download ro_core_news_sm")

    def clean_summary(self, summary_points: list[str]) -> str:
        """
        Cleans and merges a list of summary bullet points into a single text.

        Args:
            summary_points (list[str]): List of raw summary strings from the scraper.

        Returns:
            str: A single cleaned summary string.
        """

        clean_lines: list[str] = []
        for line in summary_points:
            cleaned = re.sub(r"^\d+(\.\d+)*\s*", "", line)
            cleaned = cleaned.strip().rstrip(";")

            if len(cleaned) > 10:
                clean_lines.append(cleaned)

        return " ".join(clean_lines)

    def clean_transcript(self, text: str) -> str:
        """
        Removes procedural noise and boilerplate from the transcript.

        Args:
            text (str): Raw transcript text.

        Returns:
            str: Cleaned text without applause, vociferations, or extra whitespace.
        """

        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def segment_sentences(self, text: str) -> list[str]:
        """
        Segments a long text into a list of sentences using Spacy.

        Args:
            text (str): The input text to segment.

        Returns:
            list[str]: A list of valid sentences (length > 5 chars).
        """

        doc = self.nlp(text[:100000])
        sentences = [
            sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5 # type: ignore
        ]
        return sentences

    def process_file(self, input_path: str, output_path: str) -> None:
        """
        Processes a raw JSON dataset and saves the cleaned version.

        Args:
            input_path (str): Path to the raw JSON file.
            output_path (str): Path where the processed JSON should be saved.
        """

        input_file = Path(input_path)
        if not input_file.exists():
            logging.error(f"File not found: {input_path}")
            return

        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        processed_data: list[dict[str, Any]] = []

        print(f"Processing {len(raw_data)} sessions from {input_file.name}...")

        for item in tqdm(raw_data):
            clean_summary_text = self.clean_summary(item.get("summary_points", []))
            clean_transcript_text = self.clean_transcript(
                item.get("full_transcript", "")
            )

            if not clean_summary_text or not clean_transcript_text:
                continue

            transcript_sents = self.segment_sentences(clean_transcript_text)

            processed_entry = {
                "id": item["session_id"],
                "date": item["date"],
                "abs_source": clean_transcript_text,
                "abs_target": clean_summary_text,
                "ext_source_sents": transcript_sents,
                "ext_target_text": clean_summary_text,
            }

            processed_data.append(processed_entry)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"Saved clean dataset to: {output_path}")


if __name__ == "__main__":
    from src.config import RAW_FILE, CLEAN_FILE
    processor = DataProcessor()
    processor.process_file(str(RAW_FILE), str(CLEAN_FILE))

