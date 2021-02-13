import argparse
import os
import re
import glob

# configs
from tools.config import cfg_from_yaml_file, cvt_tokenizer
from tokenizers import normalizers

# utils
from tools.utils import (
    multiprocessing_with_async,
    preprocess_mecab_pool, preprocess_shuf_pool,
    preprocess_mecab_pool_line
)

import logging
log = logging.getLogger(__name__)


def sampling(data_path: str, sample_rate: float, save_path: str = '/samples/') -> None:
    files = glob.glob(str(data_path))
    params = {'inputs': files, 'targets': ["/".join([os.path.dirname(i), save_path, os.path.basename(i)]) for i in files]}
    params.update({'sample_rate': sample_rate})
    log.debug(params)
    res = multiprocessing_with_async(params, func=preprocess_shuf_pool)
    return res, str(data_path) + str(save_path)


def morphme(data_path: str, save_path: str = '/mecab/') -> None:
    files = glob.glob(str(data_path) + '/*.txt')
    params = {'inputs': files, 'targets': ["/".join([os.path.dirname(i), save_path, os.path.basename(i)]) for i in files]}
    log.debug(params)
    res = multiprocessing_with_async(params, func=preprocess_mecab_pool)
    return res, str(data_path) + str(save_path)


def morphme_lines(data_path: str, save_path: str = '/mecab/') -> None:
    files = glob.glob(str(data_path) + '/*.txt')
    params = {}
    file_lines = []
    input_files = []
    # files
    for file_idx, _file in enumerate(files):
        read_from = open(_file, "r").read().split('\n')
        input_files.append(read_from)
        for line_idx in range(len(read_from)):
            file_lines.append("{}-{}".format(file_idx, line_idx))

    params.update({'inputs': file_lines, 'files': input_files})
    log.debug(params)
    res = multiprocessing_with_async(params, func=preprocess_mecab_pool_line)

    for file_idx in enumerate(files):
        write_file = "/".join([os.path.dirname(_file), save_path, os.path.basename(_file)])
        write_from = open(write_file, "w")
        for line, values in res[int(file_idx)].items():
            write_from.write("\n".join(values))

        write_from.close()

    return res, str(data_path) + str(save_path)


def main(cfg):
    config = cfg_from_yaml_file(cfg, cvt_tokenizer)

    # Sampling
    log.info('start sampling')
    _, path = sampling(data_path=config['Path']['data-path'], sample_rate=config['Samples']['rate'], save_path='/samples/')

    # Morphme
    if config['Morpheme-aware']:
        log.info('start morpheme')
        _, save_path = morphme(data_path=config['Path']['save-path'], save_path='/mecab/')
    else:
        save_path = config['Path']['save-path']

    texts = glob.glob(save_path + '*.txt')
    log.debug(texts)

    # tokenizer
    log.info('set tokenizer')
    tokenizer = config['Pipelines']['Tokenizer']
    tokenizer.pre_tokenizer = config['Pipelines']['pre_tokenizer']
    tokenizer.normalizer = normalizers.Sequence(config['Pipelines']['normalizer'])
    tokenizer.decoder = config['Pipelines']['decoder']

    # train
    log.info('train tokenizer')
    tokenizer.train(texts, show_progress=True)

    # eval
    log.info('eval tokenizer')
    log.info(tokenizer.encode("안녕하세요").tokens)

    # save
    os.makedirs(config['Save']['folder'], exist_ok=True)
    tokenizer.save(config['Save']['folder'] + config['Save']['file'])


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
