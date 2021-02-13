from tokenizers import normalizers


def test_tokenizer(tokenizer_cfg):
    tokenizer = tokenizer_cfg['Pipelines']['Tokenizer']
    tokenizer.pre_tokenizer = tokenizer_cfg['Pipelines']['pre_tokenizer']
    tokenizer.normalizer = normalizers.Sequence(tokenizer_cfg['Pipelines']['normalizer'])
    tokenizer.decoder = tokenizer_cfg['Pipelines']['decoder']

    tokenizer.train_from_iterator(['안녕하세요'])
    assert tokenizer.encode('안녕').tokens == ['안', '##녕']