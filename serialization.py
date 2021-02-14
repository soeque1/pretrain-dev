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

import pyarrow.parquet as pq
import pandas as pd


import logging
import traceback
log = logging.getLogger(__name__)


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

            serial = {}
            for idx, sentence in enumerate(dataset[0]):
                encoded = tokenizer.encode(str(sentence)).ids
                if len(encoded) != seq_len:
                    fail.add(file_idx)
                serial.update({idx: encoded})

            # pd
            df = pd.DataFrame.from_dict(serial, orient='columns')
            # pa
            tb = pa.Table.from_pandas(df)
            # parquet
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


def main(serializaiton_cfg):
    config = cfg_from_yaml_file(serializaiton_cfg, cvt_serialization)

    # make dir
    os.makedirs(config['Path']['save-path'], exist_ok=True)

    _, save_path = serialize_pyarrow(data_path=config['Path']['data-path'],
        save_path=config['Path']['save-path'],
        tokenizer_path=config['Tokenizer']['token'],
        tokenizer_length=config['Tokenizer']['seq_len'])


if __name__ == "__main__":
    main(serializaiton_cfg="./cfgs/serialization/pyarrow_v1.yaml")
