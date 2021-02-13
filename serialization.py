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


def main(serializaiton_cfg):
    config = cfg_from_yaml_file(serializaiton_cfg, cvt_serialization)

    # make dir
    os.makedirs(config['Path']['save-path'], exist_ok=True)

    filename = glob.glob(config['Path']['data-path'])[0]
    dataset = csv.read_csv(filename, parse_options=csv.ParseOptions(delimiter='\a', newlines_in_values='\n'))
    tokenizer = Tokenizer.from_file(config['Path']['token'])
    tokenizer.enable_padding(length=1024)
    tokenizer.enable_truncation(1024, stride=0, strategy='longest_first')

    save_file = ["/".join([config['Path']['save-path'], os.path.basename(i)]) for i in [filename]][0]
    save_file = save_file.replace('txt', 'parquet')
    serial = {}
    for idx, sentence in enumerate(dataset[0]):
        encoded = tokenizer.encode(str(sentence)).ids
        if len(encoded) != 1024:
            import pdb; pdb.set_trace()
        serial.update({idx: encoded})

    import pyarrow.parquet as pq
    import pandas as pd
    df = pd.DataFrame.from_dict(serial, orient='columns')
    tb = pa.Table.from_pandas(df)
    pq.write_table(tb, save_file)


if __name__ == "__main__":
    main(serializaiton_cfg="./cfgs/serialization/pyarrow_v1.yaml")
