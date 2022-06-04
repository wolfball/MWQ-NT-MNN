# MWQ-NT-MNN
SJTU 2021-2022-2 AI3610 Brain-Inspired Intelligence Final Project

Members: Hongwei Tu, Han Yan, Jinghao Feng

## Running commands

Run the following command for  baseline

```bash
python main.py --cuda 0 --expname exp/baseline/
```

Run the following command for our MWQ-NT-MNN

```bash
python main_improved.py --cuda 0 --expname exp/mwq_nt_mnn/
```

## Controllable args for main_improved.py

```bash
--n_values 7  # the maximum of quantized order
--mean 0.0  # the mean of noise
--std 1.0  # the std of noise (0.1~1.5 recommanded)
--quantization_threshold 0.01  # quantization threshold
--quantized_value 0.01  # quantized value
--grad_threshold 0.5  # gradient clip threshold
--cuad 0  # cuda id for running
--expname exp  # path to save
--batch_size 64
--epochs 14
--lr 1.0
--gamma 0.7  # learning rate step gamma
```

## Result Summary

