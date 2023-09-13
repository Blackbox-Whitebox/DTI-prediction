cd ../src

nohup python train.py --device cuda --config ../config/transformer/crosstransformer_classification_baseline.yml > crosstransformer_classification_baseline.log &
nohup python train.py --device cuda --config ../config/transformer/crosstransformer_contrastive_baseline.yml > crosstransformer_contrastive_baseline.log &
nohup python train.py --device cuda --config ../config/transformer/transformer_classification_baseline.yml > transformer_classification_baseline.log &