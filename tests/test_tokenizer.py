from tokenizers import normalizers


def test_tokenizer(cfg):
    tokenizer = cfg['Pipelines']['Tokenizer']
    tokenizer.pre_tokenizer = cfg['Pipelines']['pre_tokenizer']
    tokenizer.normalizer = normalizers.Sequence(cfg['Pipelines']['normalizer'])
    tokenizer.decoder = cfg['Pipelines']['decoder']

    tokenizer.train_from_iterator(['안녕하세요'])
    assert tokenizer.encode('안녕').tokens == ['안', '##녕']