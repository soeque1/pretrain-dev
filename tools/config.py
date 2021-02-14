import yaml

# tokenizer-pre_tokenizers
from tokenizers.pre_tokenizers import (
    PreTokenizer,
    ByteLevel,
    Whitespace,
    WhitespaceSplit,
    BertPreTokenizer,
    Metaspace,
    CharDelimiterSplit,
    Punctuation,
    Sequence,
    Digits,
    UnicodeScripts,
    Split,
)

# tokenizer-normalizers
from tokenizers.normalizers import NFKC, Lowercase

# tokenizers
from tools.word_piece import WordPieceTokenizer
from tools.BBPE import ByteLevelBPETokenizer
from tools.CBPE import CharBPETokenizer

from typing import Callable

from logging import getLogger

log = getLogger(__name__)


def cvt_tokenizer(config):
    config['Pipelines']['Tokenizer'] = eval(config['Pipelines']['Tokenizer'])
    config['Pipelines']['normalizer'] = [eval(i) for i in config['Pipelines']['normalizer']]
    config['Pipelines']['pre_tokenizer'] = eval(config['Pipelines']['pre_tokenizer'])
    config['Pipelines']['decoder'] = eval(config['Pipelines']['decoder'])
    return config


def cvt_serialization(config):
    return config


def cfg_from_yaml_file(cfg_file: str, func: callable) -> dict:
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    config = func(config)
    log.info(config)

    return config
