import glob
import os
from train_tokenizer import sampling, morphme

def test_samplings(cfg):
    def tear_down(cfg):
        for i in glob.glob(str(cfg['Path']['save-path']) + '/**/*.txt', recursive=True):
            os.remove(i)

    sampling(data_path=cfg['Path']['data-path'], sample_rate=cfg['Samples']['rate'], save_path='/samples/')

    if cfg['Morpheme-aware']:
        morphme(data_path=cfg['Path']['save-path'], save_path='/mecab/')

    tear_down(cfg)
