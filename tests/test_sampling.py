import glob
import os
from train_tokenizer import sampling, morphme

def test_samplings(tokenizer_cfg):
    (succ, fail), path = sampling(data_path=tokenizer_cfg['Path']['data-path'], sample_rate=tokenizer_cfg['Samples']['rate'], save_path='/samples/')
    assert len(succ) == len(glob.glob(str(tokenizer_cfg['Path']['data-path'])))
    assert len(fail) == 0

    if tokenizer_cfg['Morpheme-aware']:
        (succ, fail), path = morphme(data_path=tokenizer_cfg['Path']['save-path'], save_path='/mecab/')
        assert len(succ) == len(glob.glob(str(tokenizer_cfg['Path']['save-path']) + '/mecab/*.txt'))
        assert len(fail) == 0
