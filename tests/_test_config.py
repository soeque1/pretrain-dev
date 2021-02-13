# from tools.config import cfg_from_yaml_file


# def test_configs(cfg):
#     assert list(cfg.keys()) == ['Samples', 'Morpheme-aware', 'Path', 'Pipelines']
#     assert list(cfg.get('Samples').keys()) == ['rate']
#     assert cfg.get('Morpheme-aware') == True
#     assert list(cfg.get('Path').keys()) == ['data-path', 'save-path']
#     assert list(cfg.get('Pipelines').keys()) == ['Tokenizer', 'normalizer', 'pre_tokenizer', 'decoder']


from tools.BBPE import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import ByteLevel


from hydra.experimental import initialize, compose
def test_with_initialize() -> None:
    with initialize(config_path="../cfgs/pipelines"):
        # config is relative to a module
        cfg = compose(config_name="bbpe", overrides=["Pipelines.normalizer._target_=[]"])
        breakpoint()

        assert cfg == {
            "Samples": {"rate": 0.01},
            "Morpheme-aware": False,
            "Path": {
                "data-path": "./data/corpus/namuwiki/namuwiki.*.txt",
                "save-path": "./data/corpus/namuwiki/samples/"
            },
            "Pipelines": {
                "Tokenizer": ByteLevelBPETokenizer,
                "normalizer": [],
                "pre_tokenizer": ByteLevel,
                "decoder": ByteLevel
            }
        }