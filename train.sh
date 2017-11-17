python -m app --train \
--model_type resnet_adam \
--batch_size 25 \
2>&1 | tee >(nc seashells.io 1337)