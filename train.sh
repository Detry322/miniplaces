python -m app --train \
--model_type last_shot_resnet5 \
--batch_size 80 \
--full_size 128 --crop_size 112 \
2>&1 | tee >(nc seashells.io 1337)
python -m app --evaluate \
--model_type last_shot_resnet5 \
--batch_size 80 \
--full_size 128 --crop_size 112 
