
# Sentiment clasiffication


## Results last-linear-layer

```bash
python classifier.py --fine-tune-mode last-linear-layer --batch_size 64 --lr 1e-3 --epochs 10 --use_gpu
```

SST

```
Epoch 0: train loss :: 1.849, train acc :: 0.422, dev acc :: 0.417
Epoch 1: train loss :: 1.436, train acc :: 0.423, dev acc :: 0.421
Epoch 2: train loss :: 1.392, train acc :: 0.370, dev acc :: 0.360
Epoch 3: train loss :: 1.363, train acc :: 0.463, dev acc :: 0.442
Epoch 4: train loss :: 1.346, train acc :: 0.438, dev acc :: 0.422
Epoch 5: train loss :: 1.330, train acc :: 0.451, dev acc :: 0.434
Epoch 6: train loss :: 1.322, train acc :: 0.473, dev acc :: 0.447
Epoch 7: train loss :: 1.324, train acc :: 0.492, dev acc :: 0.453
Epoch 8: train loss :: 1.327, train acc :: 0.475, dev acc :: 0.465
Epoch 9: train loss :: 1.309, train acc :: 0.493, dev acc :: 0.452
```

cfimdb
```
Epoch 0: train loss :: 0.730, train acc :: 0.805, dev acc :: 0.812
Epoch 1: train loss :: 0.564, train acc :: 0.830, dev acc :: 0.784
Epoch 2: train loss :: 0.512, train acc :: 0.574, dev acc :: 0.563
Epoch 3: train loss :: 0.520, train acc :: 0.787, dev acc :: 0.743
Epoch 4: train loss :: 0.485, train acc :: 0.660, dev acc :: 0.673
Epoch 5: train loss :: 0.459, train acc :: 0.855, dev acc :: 0.824
Epoch 6: train loss :: 0.463, train acc :: 0.810, dev acc :: 0.804
Epoch 7: train loss :: 0.468, train acc :: 0.702, dev acc :: 0.710
Epoch 8: train loss :: 0.471, train acc :: 0.866, dev acc :: 0.861
Epoch 9: train loss :: 0.445, train acc :: 0.876, dev acc :: 0.865
```

## Results full-model


```bash
python classifier.py --fine-tune-mode full-model --batch_size 8 --lr 1e-5 --epochs 10 --use_gpu
```

SST

```
Epoch 0: train loss :: 1.571, train acc :: 0.487, dev acc :: 0.452
Epoch 1: train loss :: 1.212, train acc :: 0.568, dev acc :: 0.502
Epoch 2: train loss :: 1.089, train acc :: 0.606, dev acc :: 0.489
Epoch 3: train loss :: 1.000, train acc :: 0.652, dev acc :: 0.494
Epoch 4: train loss :: 0.915, train acc :: 0.701, dev acc :: 0.493
Epoch 5: train loss :: 0.838, train acc :: 0.756, dev acc :: 0.493
Epoch 6: train loss :: 0.757, train acc :: 0.788, dev acc :: 0.488
Epoch 7: train loss :: 0.674, train acc :: 0.827, dev acc :: 0.487
Epoch 8: train loss :: 0.589, train acc :: 0.857, dev acc :: 0.478
Epoch 9: train loss :: 0.531, train acc :: 0.880, dev acc :: 0.474
```

```bash
python classifier.py --fine-tune-mode full-model --batch_size 8 --lr 1e-5 --epochs 10 --use_gpu --hidden_dropout_prob 0.4
```


Epoch 0: train loss :: 1.622, train acc :: 0.469, dev acc :: 0.449
Epoch 1: train loss :: 1.245, train acc :: 0.557, dev acc :: 0.501
Epoch 2: train loss :: 1.112, train acc :: 0.587, dev acc :: 0.487
Epoch 3: train loss :: 1.017, train acc :: 0.642, dev acc :: 0.484


```bash
python classifier.py --fine-tune-mode full-model --batch_size 8 --lr 1e-5 --epochs 10 --use_gpu --hidden_dropout_prob 0.5
```

Epoch 0: train loss :: 1.685, train acc :: 0.440, dev acc :: 0.428
Epoch 1: train loss :: 1.286, train acc :: 0.539, dev acc :: 0.489
Epoch 2: train loss :: 1.141, train acc :: 0.581, dev acc :: 0.490
Epoch 3: train loss :: 1.049, train acc :: 0.625, dev acc :: 0.494
Epoch 4: train loss :: 0.966, train acc :: 0.675, dev acc :: 0.509
Epoch 5: train loss :: 0.892, train acc :: 0.726, dev acc :: 0.498
Epoch 6: train loss :: 0.809, train acc :: 0.769, dev acc :: 0.505
Epoch 7: train loss :: 0.740, train acc :: 0.798, dev acc :: 0.480
Epoch 8: train loss :: 0.655, train acc :: 0.803, dev acc :: 0.473
Epoch 9: train loss :: 0.593, train acc :: 0.863, dev acc :: 0.495


```bash
python classifier.py --fine-tune-mode full-model --batch_size 4 --lr 1e-5 --epochs 10 --use_gpu
```

cfimdb

```
Epoch 0: train loss :: 0.447, train acc :: 0.978, dev acc :: 0.976
Epoch 1: train loss :: 0.111, train acc :: 0.988, dev acc :: 0.951
Epoch 2: train loss :: 0.056, train acc :: 0.992, dev acc :: 0.947
Epoch 3: train loss :: 0.043, train acc :: 0.998, dev acc :: 0.967
Epoch 4: train loss :: 0.028, train acc :: 0.998, dev acc :: 0.963
Epoch 5: train loss :: 0.026, train acc :: 0.998, dev acc :: 0.959
Epoch 6: train loss :: 0.037, train acc :: 0.999, dev acc :: 0.967
Epoch 7: train loss :: 0.013, train acc :: 0.998, dev acc :: 0.967
Epoch 8: train loss :: 0.019, train acc :: 0.995, dev acc :: 0.955
Epoch 9: train loss :: 0.014, train acc :: 0.999, dev acc :: 0.967
```


