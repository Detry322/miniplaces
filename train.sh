python -m app --train \
--model_type resnet50_dropout_rmsprop_cont \
--batch_size 25 \
2>&1 | tee >(nc seashells.io 1337)