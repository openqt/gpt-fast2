import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
from typing import Dict, Optional, List

class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

class HFTokenizerWrapper(TokenizerInterface):
    """Load HuggingFace tokenizer.json (BPE/Unigram) via the tokenizers library."""
    def __init__(self, model_path):
        super().__init__(model_path)
        from tokenizers import Tokenizer
        self.processor = Tokenizer.from_file(str(model_path))
        # Qwen2 uses <|im_start|> as BOS, <|im_end|> as EOS
        self._bos_id = self._get_token_id("<|im_start|>")
        self._eos_id = self._get_token_id("<|im_end|>")

    def _get_token_id(self, token: str) -> int:
        tid = self.processor.token_to_id(token)
        if tid is not None:
            return tid
        # fallback: try encoded value
        encoded = self.processor.encode(token)
        return encoded.ids[0] if encoded.ids else 0

    def encode(self, text):
        return self.processor.encode(text).ids

    def decode(self, tokens):
        return self.processor.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


class QwenTiktokenWrapper(TokenizerInterface):
    """Qwen2.5 tiktoken-based tokenizer (qwen.tiktoken)."""
    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)

        # Qwen2.5 special tokens — fit within vocab_size - base tokens
        special_tokens = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self._bos_id = self.special_tokens["<|im_start|>"]
        self._eos_id = self.special_tokens["<|im_end|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


def get_tokenizer(tokenizer_model_path, model_name):
    """
    Factory function to get the appropriate tokenizer based on the model name.

    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """

    model_str = str(model_name).lower()
    tokenizer_path = Path(tokenizer_model_path)

    # Qwen2.5 uses tiktoken with qwen.tiktoken
    if "qwen2.5" in model_str or "qwen-2.5" in model_str:
        # Prefer tokenizer.json (has complete special tokens)
        hf_tokenizer = tokenizer_path.parent / "tokenizer.json"
        if hf_tokenizer.exists():
            return HFTokenizerWrapper(hf_tokenizer)
        qwen_tiktoken = tokenizer_path.parent / "qwen.tiktoken"
        if qwen_tiktoken.exists():
            return QwenTiktokenWrapper(qwen_tiktoken)

    # Qwen2 uses BPE via tokenizers library
    if "qwen2" in model_str or "qwen-2" in model_str:
        hf_tokenizer = tokenizer_path.parent / "tokenizer.json"
        if hf_tokenizer.exists():
            return HFTokenizerWrapper(hf_tokenizer)
        # fallback: try standard files
        if tokenizer_path.exists():
            return SentencePieceWrapper(tokenizer_path)

    # LLaMA 3+ uses tiktoken
    if "llama-3" in model_str:
        return TiktokenWrapper(tokenizer_model_path)

    # Default: SentencePiece (LLaMA 1/2, Qwen fallback, etc.)
    if tokenizer_path.exists():
        return SentencePieceWrapper(tokenizer_model_path)

    # Last resort: try tokenizer.json with HF tokenizers
    hf_tokenizer = tokenizer_path.parent / "tokenizer.json"
    if hf_tokenizer.exists():
        return HFTokenizerWrapper(hf_tokenizer)

    raise FileNotFoundError(f"No tokenizer found at {tokenizer_model_path}")
