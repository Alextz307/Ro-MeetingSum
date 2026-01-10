import torch
import spacy
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding

from src.models.matchsum_ro import RomanianMatchSum
from src.config import EXTRACTIVE_MODEL_PATH, BASE_BERT_MODEL


class ExtractiveSummarizer:
    """
    Performs specific extractive summarization using a Siamese BERT model (MatchSum).
    """

    def __init__(self, checkpoint_path: str):
        """
        Loads the RomanianMatchSum model and tokenizer.

        Args:
            checkpoint_path (str): Path to the saved model weights (.pt file).
        """

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Loading Model on {self.device}...")

        self.model = RomanianMatchSum(BASE_BERT_MODEL)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            BASE_BERT_MODEL
        )
        try:
            self.nlp = spacy.load("ro_core_news_sm")
        except OSError:
            print("Spacy model not found. Running download...")
            from spacy.cli import download  # type: ignore

            download("ro_core_news_sm")
            self.nlp = spacy.load("ro_core_news_sm")

    def _tokenize(
        self, text: str, max_len: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper to tokeniz text for BERT."""

        encoded: BatchEncoding = self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(self.device),  # type: ignore
            encoded["attention_mask"].to(self.device),  # type: ignore
        )

    def summarize(
        self, text: str, ratio: float = 0.2, min_k: int = 3, max_k: int = 15
    ) -> str:
        """
        Extracts key sentences from the text based on semantic similarity to the whole document.

        Args:
            text (str): Input text to summarize.
            ratio (float): Fraction of sentences to extract (0.0 to 1.0).
            min_k (int): Minimum number of sentences to return.
            max_k (int): Maximum number of sentences to return.

        Returns:
            str: The extracted summary composed of the top ranked sentences.
        """

        doc = self.nlp(text)
        sentences = [
            sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10
        ]

        total_sentences = len(sentences)
        if not sentences:
            return "Text prea scurt pentru rezumat."

        calculated_k = int(total_sentences * ratio)
        top_k = max(min_k, min(max_k, calculated_k))
        top_k = min(top_k, total_sentences)

        print(
            f"Input Sentences: {total_sentences} | Selecting Top {top_k} ({ratio*100}%)"
        )

        doc_text = " ".join(sentences)[:3000]
        doc_ids, doc_mask = self._tokenize(doc_text)

        with torch.no_grad():
            doc_emb = self.model(doc_ids, doc_mask)

        scores: list[tuple[float, str]] = []

        for sent in sentences:
            if len(sent) > 500:
                continue

            sent_ids, sent_mask = self._tokenize(sent, max_len=128)
            with torch.no_grad():
                sent_emb = self.model(sent_ids, sent_mask)
                score = torch.nn.functional.cosine_similarity(doc_emb, sent_emb).item()
                scores.append((score, sent))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scores[:top_k]

        top_sentences_ordered = sorted(
            top_sentences, key=lambda x: sentences.index(x[1])
        )

        final_summary = " ".join([s[1] for s in top_sentences_ordered])
        return final_summary


if __name__ == "__main__":
    current_path = str(EXTRACTIVE_MODEL_PATH)
    if not Path(current_path).exists():
        print(f"Model not found at {current_path}.")
    else:
        summarizer = ExtractiveSummarizer(current_path)
        sample_text = (
            "Astăzi discutăm despre buget. Este inadmisibil să nu avem fonduri."
        )
        summary = summarizer.summarize(sample_text)
        print(summary)
