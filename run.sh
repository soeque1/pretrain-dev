# pre-process
rm -rf data/corpus/namuwiki/samples/
mkdir -p data/corpus/namuwiki/samples/mecab/

# run
python train_tokenizer.py \
    --cfg=./cfgs/pipelines/word_piece_with_morpheme.yaml

python train_tokenizer.py \
    --cfg=./cfgs/pipelines/word_piece.yaml

python train_tokenizer.py \
    --cfg=./cfgs/pipelines/cbpe.yaml

python train_tokenizer.py \
    --cfg=./cfgs/pipelines/bbpe.yaml
