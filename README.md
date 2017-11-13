# miniplaces
6.869 Miniplaces Project

### Getting set up

Run:

```
pip install -r requirements.txt
python -m app --download
```

If you're running on linux and want to train on GPUs, also do:

```
pip install tensorflow-gpu
```

### Training the model

Run:

```
python -m app --train --model_type basic_model 2>&1 | tee >(nc seashells.io 1337)
```

You'll get a link which you can send to friends which shows the training progress.

