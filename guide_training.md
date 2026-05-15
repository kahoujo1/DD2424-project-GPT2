
# Sentiment clasiffication

```bash
python classifier.py --fine-tune-mode last-linear-layer --batch_size 64 --lr 1e-3 --epochs 10 --use_gpu
python classifier.py --fine-tune-mode full-model --batch_size 8 --lr 1e-5 --epochs 10 --use_gpu
```

With Lora:

```bash
python classifier.py --fine-tune-mode last-linear-layer --batch_size 64 --lr 1e-3 --epochs 10 --use_gpu --enable_lora
```


# Sonnet generation:

```bash
python sonnet_generation.py --lr 1e-5 --epochs 8 --use_gpu --model_size gpt2


python sonnet_generation.py --lr 1e-3 --epochs 64 --use_gpu --model_size gpt2 --enable_lora
```

