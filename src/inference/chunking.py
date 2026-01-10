import math
from typing import Any


def summarize_in_chunks(summ_pipeline: Any, long_text: str, tokenizer: Any) -> str:
    """
    Splits a long text into chunks and summarizes them individually.
    This avoids model input size limits (e.g. 512 or 1024 tokens).

    Args:
        summ_pipeline (Any): The HuggingFace summarization pipeline.
        long_text (str): The long input text.
        tokenizer (Any): Tokenizer associated with the model (to count tokens).

    Returns:
        str: A concatenated string of chunk summaries.
    """

    inputs = tokenizer(long_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]
    total_tokens = len(input_ids)

    stride = 450
    num_chunks = math.ceil(total_tokens / stride)

    final_summary_parts: list[str] = []

    for i in range(num_chunks):
        start = i * stride
        end = min((i + 1) * stride, total_tokens)
        chunk_ids = input_ids[start:end]

        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        if len(chunk_text.strip()) < 50:
            continue

        try:
            gen = summ_pipeline(
                "summarize: " + chunk_text,
                max_length=300,
                min_length=60,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                early_stopping=True,
                truncation=True,
            )
            part = gen[0]["summary_text"]
            final_summary_parts.append(part)
        except Exception:
            pass

    return " ".join(final_summary_parts)
