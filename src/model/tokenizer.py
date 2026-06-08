from transformers import GPT2Tokenizer


_tokenizer = None


def get_tokenizer() -> GPT2Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer
