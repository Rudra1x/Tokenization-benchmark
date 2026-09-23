"""
Word-level tokenizer implemented from scratch.

This tokenizer:
    1. Normalizes input text.
    2. Splits text into word/punctuation tokens.
    3. Builds a vocabulary from training data.
    4. Maps tokens <-> integer IDs.
    5. Uses <UNK> for unseen tokens.

No external tokenization library is used.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence


class WordTokenizer:
    """
    Simple word-level tokenizer.

    The vocabulary must be learned using training data only.

    Special tokens:
        <PAD> : padding token
        <UNK> : unknown token
        <BOS> : beginning-of-sequence token
        <EOS> : end-of-sequence token
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        self.vocab_size: int = 0
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text before tokenization.

        We intentionally keep normalization conservative because this is
        a multilingual benchmark. Unicode characters and casing are
        preserved.

        Only repeated whitespace is normalized.
        """
        return re.sub(r"\s+", " ", text.strip())

    @classmethod
    def tokenize_text(cls, text: str) -> List[str]:
        """
        Split text into word and punctuation tokens.

        Examples:
            "Hello world!" -> ["Hello", "world", "!"]

            "Machine-learning" ->
                ["Machine", "-", "learning"]

        Unicode word characters are supported through Python's regex
        implementation.
        """
        text = cls.normalize(text)

        if not text:
            return []

        # \w handles Unicode word characters in Python.
        # \S catches non-whitespace punctuation/symbols.
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: Iterable[str],
        min_frequency: int = 1,
    ) -> "WordTokenizer":
        """
        Build vocabulary from training texts.

        Parameters
        ----------
        texts:
            Training corpus only.

        min_frequency:
            Minimum number of occurrences required for a token
            to enter the vocabulary.
        """

        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1.")

        counter = Counter()

        for text in texts:
            counter.update(self.tokenize_text(text))

        # Special tokens always occupy the first IDs.
        self.token_to_id = {
            token: idx
            for idx, token in enumerate(self.SPECIAL_TOKENS)
        }

        # Sort by:
        #   1. frequency descending
        #   2. token alphabetically for deterministic results
        vocabulary_tokens = sorted(
            (
                token
                for token, frequency in counter.items()
                if frequency >= min_frequency
                and token not in self.token_to_id
            ),
            key=lambda token: (-counter[token], token),
        )

        for token in vocabulary_tokens:
            self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

        self.vocab_size = len(self.token_to_id)
        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """
        Convert text into token IDs.

        Unknown tokens are mapped to <UNK>.
        """

        self._check_fitted()

        tokens = self.tokenize_text(text)

        token_ids = [
            self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"],
            )
            for token in tokens
        ]

        if add_special_tokens:
            token_ids = (
                [self.token_to_id["<BOS>"]]
                + token_ids
                + [self.token_to_id["<EOS>"]]
            )

        return token_ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Convert token IDs back into a readable string.
        """

        self._check_fitted()

        tokens = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            token = self.id_to_token[token_id]

            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue

            tokens.append(token)

        # Reconstruct readable text.
        text = " ".join(tokens)

        # Remove spaces before common punctuation.
        text = re.sub(r"\s+([,.!?;:%)\]}])", r"\1", text)

        # Remove spaces after opening brackets.
        text = re.sub(r"([(\[{])\s+", r"\1", text)

        return text

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        """
        Public tokenization method.
        """
        return self.tokenize_text(text)

    def token_id(self, token: str) -> int:
        """
        Return the ID of a token.

        Raises KeyError if the token is not in vocabulary.
        """
        self._check_fitted()

        return self.token_to_id[token]

    def token(self, token_id: int) -> str:
        """
        Return the token corresponding to an ID.
        """
        self._check_fitted()

        if token_id not in self.id_to_token:
            raise ValueError(
                f"Unknown token ID: {token_id}"
            )

        return self.id_to_token[token_id]

    def get_vocab(self) -> dict[str, int]:
        """
        Return a copy of the vocabulary.
        """
        self._check_fitted()

        return self.token_to_id.copy()

    # ------------------------------------------------------------------
    # OOV handling
    # ------------------------------------------------------------------

    def count_oov(self, text: str) -> int:
        """
        Count the number of OOV tokens in a text.

        Important:
            OOV is measured against the vocabulary learned during fit().
        """

        self._check_fitted()

        tokens = self.tokenize_text(text)

        return sum(
            token not in self.token_to_id
            for token in tokens
        )

    def oov_rate(self, texts: Iterable[str]) -> float:
        """
        Calculate token-level OOV rate over a collection of texts.

        OOV rate =
            number of OOV tokens / total tokens
        """

        self._check_fitted()

        total_tokens = 0
        total_oov = 0

        for text in texts:
            tokens = self.tokenize_text(text)

            total_tokens += len(tokens)

            total_oov += sum(
                token not in self.token_to_id
                for token in tokens
            )

        if total_tokens == 0:
            return 0.0

        return total_oov / total_tokens

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """
        Ensure that the tokenizer has been fitted before encoding,
        decoding, or calculating OOV.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Tokenizer has not been fitted. "
                "Call fit() using training data first."
            )

    def __len__(self) -> int:
        """
        Return vocabulary size.
        """
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"WordTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"fitted={self.is_fitted})"
        )