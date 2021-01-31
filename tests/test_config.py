from tools.config import cfg_from_yaml_file


def test_configs(cfg):
    assert list(cfg.keys()) == ['Samples', 'Morpheme-aware', 'Path', 'Pipelines']
    assert list(cfg.get('Samples').keys()) == ['rate']
    assert cfg.get('Morpheme-aware') == True
    assert list(cfg.get('Path').keys()) == ['data-path', 'save-path']
    assert list(cfg.get('Pipelines').keys()) == ['Tokenizer', 'normalizer', 'pre_tokenizer', 'decoder']
