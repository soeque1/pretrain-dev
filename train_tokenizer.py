import argparse
import os
import re
import glob
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
from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Lowercase

# tokenizers
from tools.word_piece import WordPieceTokenizer
from tools.BBPE import ByteLevelBPETokenizer
from tools.CBPE import CharBPETokenizer

# utils
from tools.utils import (
    multiprocessing_with_async,
    preprocess_mecab_pool, preprocess_shuf_pool
)



def cfg_from_yaml_file(cfg_file):
    def check_and_evalfunc(config):
        config['Pipelines']['Tokenizer'] = eval(config['Pipelines']['Tokenizer'])
        config['Pipelines']['normalizer'] = [eval(i) for i in config['Pipelines']['normalizer']]
        config['Pipelines']['pre_tokenizer'] = eval(config['Pipelines']['pre_tokenizer'])
        config['Pipelines']['decoder'] = eval(config['Pipelines']['decoder'])
        return config

    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    config = check_and_evalfunc(config)

    return config


def main(cfg):
    config = cfg_from_yaml_file(cfg)
    print(config)

    # Sampling
    files = glob.glob(config['Path']['data-path'])
    params = {'inputs': files, 'targets': ["/".join([os.path.dirname(i), '/samples/', os.path.basename(i)]) for i in files]}
    params.update({'sample_rate': config['Samples']['rate']})
    multiprocessing_with_async(params, func=preprocess_shuf_pool)

    # Morphme
    if config['Morpheme-aware']:
        files = glob.glob(config['Path']['save-path'] + '/*.txt')
        params = {'inputs': files, 'targets': ["/".join([os.path.dirname(i), '/mecab/', os.path.basename(i)]) for i in files]}
        multiprocessing_with_async(params, func=preprocess_mecab_pool)
        texts = glob.glob(config['Path']['save-path'] + '/mecab/*.txt')
    else:
        texts = glob.glob(config['Path']['save-path'] + '*.txt')

    # tokenizer
    tokenizer = config['Pipelines']['Tokenizer']
    tokenizer.pre_tokenizer = config['Pipelines']['pre_tokenizer']
    tokenizer.normalizer = normalizers.Sequence(config['Pipelines']['normalizer'])
    tokenizer.decoder = config['Pipelines']['decoder']

    # train
    tokenizer.train(texts, show_progress=True)

    # eval
    print(tokenizer.encode("안녕하세요").tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default=None,
        metavar="path",
        type=str,
        required=True,
        help="",
    )

    args = parser.parse_args()
    main(args.cfg)
