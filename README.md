# Noise Attention Learning: Enhancing Noise Robustness by Gradient Scaling

This repository is the implementation of **Noise Attention Learning: Enhancing Noise Robustness by Gradient Scaling** (NeurIPS 2022).

## Requirements

This codebase is written for `python3`, other necessary python packages are including

- numpy=1.20.1
- torch=1.8.1
- torchvision=0.2.2
- Pillow=8.2.0

## Training

### Usage
To train the model in the paper, run the following commands:
#### Datasets with synthetic label noise
Train the network on the CIFAR-10 dataset with 60% symmmetric label noise:
```train
python train_cifar.py --root_dir <path_to_save_data> --noise_mode 'sym' --r 0.6 
```
Train the network on the CIFAR-10 dataset with 40% asymmmetric label noise :
```train
python train_cifar.py --root_dir <path_to_save_data> --noise_mode 'asym' --r 0.4 --num_epochs 300 
```

Please refer to the Appendix for more information of hyperparameters.
```
hyperparameter options:
--batch_size            batch size
--lr                    learning rate
--num_epochs            number of epochs
--noise_mode            label noise model(e.g. sym, asym)
--r                     noise level (e.g. 0.4)
--lam                   penalty loss coefficient
--delta                 confidence threshold (optional)
--beta                  auxiliary regularization coefficient (optional for RNAL)
--m                     momentum for target estimation
--es                    epoch that starts to perform target estimation
```

#### Datasets with real-world label noise

Before running the code, please download the datasets [ANIMAL-10N](https://dm.kaist.ac.kr/datasets/animal-10n/), [Clothing1M](https://github.com/Cysu/noisy_label) and [Webvision](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html).


## Citing this work
If you use this code in your work, please cite our paper.
