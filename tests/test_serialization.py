import glob
from serialization import serialize_pyarrow, serialize_mmap


def test_pyarrow_serialization(serialization_pyarrow_cfg):
    (succ, fail), _ = serialize_pyarrow(data_path=str(serialization_pyarrow_cfg['Path']['data-path']),
        save_path=str(serialization_pyarrow_cfg['Path']['save-path']),
        tokenizer_path=serialization_pyarrow_cfg['Tokenizer']['token'],
        tokenizer_length=serialization_pyarrow_cfg['Tokenizer']['seq_len'])

    assert len(fail) == 0


def test_mmap_serialization(serialization_mmap_cfg):
    (succ, fail), _ = serialize_mmap(data_path=str(serialization_mmap_cfg['Path']['data-path']),
        save_path=str(serialization_mmap_cfg['Path']['save-path']),
        tokenizer_path=serialization_mmap_cfg['Tokenizer']['token'],
        tokenizer_length=serialization_mmap_cfg['Tokenizer']['seq_len'])

    assert len(fail) == 0