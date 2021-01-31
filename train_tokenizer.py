import argparse
import os
import re
import glob

# configs
from tools.config import cfg_from_yaml_file
from tokenizers import normalizers

# utils
from tools.utils import (
    multiprocessing_with_async,
    preprocess_mecab_pool, preprocess_shuf_pool
)


def sampling(data_path: str, sample_rate: float, save_path: str = '/samples/') -> None:
    files = glob.glob(str(data_path))
    params = {'inputs': files, 'targets': ["/".join([os.path.dirname(i), save_path, os.path.basename(i)]) for i in files]}
    params.update({'sample_rate': sample_rate})
    res = multiprocessing_with_async(params, func=preprocess_shuf_pool)
    return res, str(data_path) + str(save_path)


def morphme(data_path: str, save_path: str = '/mecab/') -> None:
    files = glob.glob(str(data_path) + '/*.txt')
    params = {'inputs': files, 'targets': ["/".join([os.path.dirname(i), save_path, os.path.basename(i)]) for i in files]}
    res = multiprocessing_with_async(params, func=preprocess_mecab_pool)
    return res, str(data_path) + str(save_path)


def main(cfg):
    config = cfg_from_yaml_file(cfg)

    # Sampling
    _, path = sampling(data_path=config['Path']['data-path'], sample_rate=config['Samples']['rate'], save_path='/samples/')

    # Morphme
    if config['Morpheme-aware']:
        _, save_path = morphme(data_path=config['Path']['save-path'], save_path='/mecab/')
    else:
        save_path = config['Path']['save-path']

    texts = glob.glob(save_path + '*.txt')

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
