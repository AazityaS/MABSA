"""Text tokenization helpers; original multilingual text is never rewritten."""


def load_xlmr_tokenizer(model_name: str = "FacebookAI/xlm-roberta-base"):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to load the XLM-R tokenizer.") from exc
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(texts, tokenizer, max_length: int = 256):
    if max_length < 1:
        raise ValueError("max_length must be positive")
    return tokenizer(list(texts), padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")

