import glob
import os
from train_tokenizer import sampling, morphme

def test_samplings(cfg):
    def tear_down(cfg):
        for i in glob.glob(str(cfg['Path']['save-path']) + '/**/*.txt', recursive=True):
            os.remove(i)

    (succ, fail), path = sampling(data_path=cfg['Path']['data-path'], sample_rate=cfg['Samples']['rate'], save_path='/samples/')
    assert len(succ) == len(glob.glob(str(cfg['Path']['data-path'])))
    assert len(fail) == 0

    if cfg['Morpheme-aware']:
        (succ, fail), path = morphme(data_path=cfg['Path']['save-path'], save_path='/mecab/')
        assert len(succ) == len(glob.glob(str(cfg['Path']['save-path']) + '/mecab/*.txt'))
        assert len(fail) == 0

    tear_down(cfg)
