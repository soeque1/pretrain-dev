import glob
from serialization import serialize_pyarrow


def test_serialization(serialization_cfg):
    (succ, fail), _ = serialize_pyarrow(data_path=str(serialization_cfg['Path']['data-path']),
        save_path=str(serialization_cfg['Path']['save-path']),
        tokenizer_path=serialization_cfg['Tokenizer']['token'],
        tokenizer_length=serialization_cfg['Tokenizer']['seq_len'])

    assert len(fail) == 0