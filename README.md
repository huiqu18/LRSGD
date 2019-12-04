# Local Regularizer Improves Generalization (LRSGD)

This repository contains the pytorch code for the paper:

Yikai Zhang*, Hui Qu*, Dimitris Metaxas, and Chao Chen, "Local Regularizer Improves Generalization", AAAI, 2020. ([PDF](https://doi.org/10.1007/978-3-030-32239-7_42))

If you find this code helpful, please cite our work:

```
@inproceedings{zhang2020local,
    author = "Yikai Zhang, Hui Qu, Dimitris Metaxas, and Chao Chen",
    title = "Local Regularizer Improves Generalization",
    booktitle = "AAAI",
    year = "2020"
}
```

## Introduction

Regularization plays an important role in generalization of deep learning. In this paper, we study the generalization
power of an unbiased regularizor for training algorithms in deep learning. We focus on training methods called Locally
Regularized Stochastic Gradient Descent (LRSGD). An LRSGD leverages a proximal type penalty in gradient descent
steps to regularize SGD in training. We show that by carefully choosing relevant parameters, LRSGD generalizes
better than SGD. Our thorough theoretical analysis is supported by experimental evidence. It advances our theoretical
understanding of deep learning and provides new perspectives on designing training algorithms.


## Usage

To training a model, set related parameters in the file `options.py` and run `python train.py`, 
or pass the parameter values from the command line
```
python train.py --optimizer lrsgdc --random-seed 1 --model ResNet18 --epochs 350 --lr 0.1 \
  --batch-size 128 --save-dir ./experiments/LRSGDC/ResNet18
```