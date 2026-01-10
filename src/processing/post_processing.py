import re


def clean_text_for_generation(text: str) -> str:
    """
    Aggressively cleans text to remove parliamentary specific boilerplate.
    Useful before feeding text into an abstractive model to avoid generating noise.

    Args:
        text (str): Input text (likely a transcript or extractive summary).

    Returns:
        str: Cleaned text.
    """

    patterns = [
        r"Deputat.*?(?=\.)",
        r"Circumscripţia.*?(?=\.)",
        r"Declaraţia.*?intitulată",
        r"Stimaţi colegi",
        r"Doamna preşedinte",
        r"Domnule preşedinte",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()
