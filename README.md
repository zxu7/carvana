## Intro
Baseline solutions to Kaggle Carvana Image Segmentation Project.

Link: https://www.kaggle.com/c/carvana-image-masking-challenge

## Dependencies
- Python 3
- numpy
- keras >= 2.0.3
- tensorflow >= 1.0.1

## Execution
```
$ python train.py
usage: python train.py [--train_imdir TRAIN_DIRECTORY] [--train_maskdir TRAIN_MASK_DIRECTORY] [--target_size IMAGE_SIZE]
                     [--grayscale GRAYSCALE] [batch_size BATCH_SIZE] [epochs NUM_EPOCH] [gpu %GPU] [gpus GPUS]

optional arguments:
    --train_imdir TRAIN_DIRECTORY           path to training images directory
    --train_maskdir TRAIN_MASK_DIRECTORY    path to training masks directory
    --target_size IMAGE_SIZE                input image size (default: (256, 256))
    --grayscale GRAYSCALE                   use grayscale (default: True)
    --batch_size BATCH_SIZE                 batch size (default: 2)
    --epochs NUM_EPOCH                      number of epochs to train for (default: 20)
    --gpu %GPU                              percentage of GPU to use (default: 1.0)
    --gpus GPUS                             which GPU to use (default: None)

$ python test.py
usage: python test.py [gpu %GPU] [gpus GPUS]

optional arguments for neural network model:
    --gpu %GPU                              percentage of GPU to use (default: 1.0)
    --gpus GPUS                             which GPU to use (default: None)
```

## RESULTS
on a NVIDIA 1180T GPU, the network was able to achieve ~0.97 train score and ~0.90 valid score