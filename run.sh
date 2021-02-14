# pre-process
rm -rf data/corpus/namuwiki/samples/
mkdir -p data/corpus/namuwiki/samples/mecab/
rm -rf data/token/namuwiki

# run
## tokenizer

python train_tokenizer.py \
    --cfg=./cfgs/pipelines/word_piece_with_morpheme.yaml

python train_tokenizer.py \
    --cfg=./cfgs/pipelines/word_piece.yaml

python train_tokenizer.py \
    --cfg=./cfgs/pipelines/cbpe.yaml

python train_tokenizer.py \
    --cfg=./cfgs/pipelines/bbpe.yaml

## serialize
python serialization.py \
    --cfg=./cfgs/serialization/mmap_v1.yaml
