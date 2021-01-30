from tokenizers import Tokenizer, AddedToken, pre_tokenizers, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import (
    Sequence,
    Lowercase,
    unicode_normalizer_from_str,
    BertNormalizer,
)
from .base_tokenizer import BaseTokenizer

from typing import Optional, List, Union, Dict, Tuple, Iterator


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece Tokenizer """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        wordpieces_prefix: str = "##",
        lowercase: bool = False,
        unicode_normalizer: Optional[str] = None,
        bert_normalizer: bool = True,
        split_on_whitespace_only: bool = False,
    ):
        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(unk_token)))
        else:
            tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token)))

        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if bert_normalizer:
            normalizers += [BertNormalizer(lowercase=False)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        if split_on_whitespace_only:
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        parameters = {
            "model": "WordPiece",
            "unk_token": unk_token,
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
            "bert_normalizer": bert_normalizer,
            "split_on_whitespace_only": split_on_whitespace_only,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab_filename: str, **kwargs):
        vocab = WordPiece.read_file(vocab_filename)
        return WordPieceTokenizer(vocab, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = ["[UNK]"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        suffix: Optional[str] = "</w>",
        show_progress: bool = True,
    ):
        """ Train the model using the given files """

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            end_of_word_suffix=suffix,
            show_progress=show_progress,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = ["[UNK]"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        suffix: Optional[str] = "</w>",
        show_progress: bool = True,
    ):
        """ Train the model using the given iterator """

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            end_of_word_suffix=suffix,
            show_progress=show_progress,
        )
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)