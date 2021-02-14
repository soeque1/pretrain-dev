from tools.config import cfg_from_yaml_file


def test_tokenizer_configs(tokenizer_cfg):
    assert list(tokenizer_cfg.keys()) == ['Samples', 'Morpheme-aware', 'Path', 'Pipelines']
    assert list(tokenizer_cfg.get('Samples').keys()) == ['rate']
    assert tokenizer_cfg.get('Morpheme-aware') == True
    assert list(tokenizer_cfg.get('Path').keys()) == ['data-path', 'save-path']
    assert list(tokenizer_cfg.get('Pipelines').keys()) == ['Tokenizer', 'normalizer', 'pre_tokenizer', 'decoder']


def test_sericalization_configs(serialization_cfg):
    assert list(serialization_cfg.keys()) == ['Path', 'Tokenizer', "Method"]
    assert list(serialization_cfg.get('Path').keys()) == ['data-path', 'save-path']
    assert list(serialization_cfg.get('Tokenizer').keys()) == ['token', 'seq_len']
    assert serialization_cfg.get('Method').get('serialization') == 'pyarrow'
