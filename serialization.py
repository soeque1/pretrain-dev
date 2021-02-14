import argparse
import glob
import os
import re

# configs
from tools.config import cfg_from_yaml_file, cvt_serialization

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv
from pyarrow import fs

from tokenizers import Tokenizer

from tools.utils import (
    multiprocessing_with_async
)

# pyarrow
import pyarrow.parquet as pq
import pandas as pd

# mmap
import torch
from megatron.data import indexed_dataset


import logging
import traceback
log = logging.getLogger(__name__)


def serialize_mmap_pool(params_index):
    """
    """
    params = params_index['params']
    idx = params_index['idx']

    succ = set()
    fail = set()

    for file_idx in range(idx, idx+1):
        try:
            # TODO: args
            level = 'sentence'
            key = 'text'

            data_file = params['inputs'][file_idx]
            output_prefix = "{}-{}".format(params['target-prefix'], file_idx)

            dataset = csv.read_csv(data_file, parse_options=csv.ParseOptions(delimiter='\a', newlines_in_values='\n'))
            tokenizer = Tokenizer.from_file(params['tokenizer_path'])
            seq_len = params['tokenizer_length']
            tokenizer.enable_padding(length=seq_len)
            tokenizer.enable_truncation(seq_len, stride=0, strategy='longest_first')

            output_bin_files = "{}_{}_{}.bin".format(output_prefix, key, level)
            output_idx_files = "{}_{}_{}.idx".format(output_prefix, key, level)

            builders = indexed_dataset.make_builder(output_bin_files, impl='mmap', vocab_size=len(tokenizer.get_vocab()))

            for idx, sentence in enumerate(dataset[0]):
                encoded = tokenizer.encode(str(sentence)).ids
                builders.add_item(torch.IntTensor(encoded))
                if len(encoded) != seq_len:
                    fail.add(file_idx)

            builders.finalize(output_idx_files)
            succ.add(file_idx)
        except Exception:
            log.error(traceback.print_exc())
            fail.add(file_idx)

    return (succ, fail)


def serialize_pyarrow_pool(params_index):
    """
    cons:
        - predefined padding tokens
        - TODO: buffer, serialization
    """
    params = params_index['params']
    idx = params_index['idx']

    succ = set()
    fail = set()

    for file_idx in range(idx, idx+1):
        try:
            data_file = params['inputs'][file_idx]
            save_file = params['targets'][file_idx]

            dataset = csv.read_csv(data_file, parse_options=csv.ParseOptions(delimiter='\a', newlines_in_values='\n'))
            tokenizer = Tokenizer.from_file(params['tokenizer_path'])
            seq_len = params['tokenizer_length']
            tokenizer.enable_padding(length=seq_len)
            tokenizer.enable_truncation(seq_len, stride=0, strategy='longest_first')

            log.info("{}-{}".format(file_idx, 'text encoding'))
            serial = {}
            for idx, sentence in enumerate(dataset[0]):
                encoded = tokenizer.encode(str(sentence)).ids
                if len(encoded) != seq_len:
                    fail.add(file_idx)
                serial.update({idx: encoded})

            log.info("{}-{}".format(file_idx, 'pandas format'))
            df = pd.DataFrame.from_dict(serial, orient='columns')

            log.info("{}-{}".format(file_idx, 'pyarrow'))
            tb = pa.Table.from_pandas(df)

            log.info("{}-{}".format(file_idx, 'parquet'))
            pq.write_table(tb, save_file)
            succ.add(file_idx)
        except Exception:
            log.error(traceback.print_exc())
            fail.add(file_idx)

    return (succ, fail)


def serialize_pyarrow(data_path: str, save_path: str, tokenizer_path: str, tokenizer_length: int) -> None:
    files = glob.glob(str(data_path))
    params = {'inputs': files, 'targets': ["/".join([save_path, os.path.basename(i).replace('txt', 'parquet')]) for i in files],
        'tokenizer_path': tokenizer_path, 'tokenizer_length': tokenizer_length}
    log.debug(params)
    res = multiprocessing_with_async(params, func=serialize_pyarrow_pool)
    return res, str(save_path)


def serialize_mmap(data_path: str, save_path: str, tokenizer_path: str, tokenizer_length: int) -> None:
    files = glob.glob(str(data_path))
    params = {'inputs': files, 'target-prefix': save_path,
        'tokenizer_path': tokenizer_path, 'tokenizer_length': tokenizer_length}
    log.debug(params)
    res = multiprocessing_with_async(params, func=serialize_mmap_pool)
    return res, str(save_path)


def main(serializaiton_cfg):
    config = cfg_from_yaml_file(serializaiton_cfg, cvt_serialization)

    # make dir
    os.makedirs(config['Path']['save-path'], exist_ok=True)

    if config['Method']['serialization'] == 'pyarrow':
        serialize = serialize_pyarrow
    elif config['Method']['serialization'] == 'mmap':
        serialize = serialize_mmap
    else:
        raise NotImplementedError

    log.info('Serialize')
    _, save_path = serialize(data_path=config['Path']['data-path'],
        save_path=config['Path']['save-path'],
        tokenizer_path=config['Tokenizer']['token'],
        tokenizer_length=config['Tokenizer']['seq_len'])


if __name__ == "__main__":
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
    main(serializaiton_cfg=args.cfg)
