# Code for Ultrasound Image Enhancement Challenge (USenhance) 2023 in MICCAI
For the validation phase, we used GFE-Net, the method we proposed in A Generic Fundus Image Enhancement Network Boosted by Frequency Self-supervised Representation Learning [[arXiv]](https://arxiv.org/abs/2206.04684). 

For the test phase, we proposed a structure-preserving medical image enhancement method based on unpaired training.

# Enhancing and Adapting in the Clinic: Test-time Adaptation for Medical Image Enhancement
We propose an algorithm for test-time adaptive medical image enhancement (TAME), which adapts and optimizes enhancement models using test data in the inference phase.


# Prerequisites

\- Win10

\- Python 3

\- CPU or NVIDIA GPU + CUDA CuDNN

# Environment (Using conda)

```
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing opencv-python

conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9

conda install visdom dominate -c conda-forge # install visdom and dominate
```

# Visualization when training

python -m visdom.server

# To open this link in the browser

http://localhost:8097/


# Command to run

Please note that root directory is the project root directory.

## Train

For SCR-Net:

```
python train.py --dataroot ./datasets/ultrasound --name train_ultrasound_stillgan_twolow --eval_test --num_test 202-- gpu_ids 5 --test_when_train --test_freq 2 --display_id 430810 --batch_size 2 --model still_gan_scr --input_nc 1 --output_nc 1 --direction AtoB --dataset_mode Ultrasound_stillgan --lr_policy linear --n_epochs 200 --n_epochs_decay 100 --test_when_train --display_port 9013 --lr 0.001 --netG unet_combine_2layer
```

For Unpaired Structure Persevere Medical Image Enhancement:

```
python train.py --dataroot ./datasets/ultrasound --name train_ultrasound_stillgan_twolow --eval_test --num_test 202-- gpu_ids 5 --test_when_train --test_freq 2 --display_id 430810 --batch_size 2 --model still_gan --input_nc 1 --output_nc 1 --direction AtoB --dataset_mode Ultrasound_stillgan --lr_policy linear --n_epochs 200 --n_epochs_decay 100 --test_when_train --display_port 9013 --lr 0.001
```


Released soon.

## Test & Visualization

For SCR-Net:

```
python test_stillgan.py --dataroot ./datasets/ultrasound --name train_ultrasound_stillgan_twolow --model still_gan_singlescr --input_nc 1 --output_nc 1 --direction AtoB --dataset_mode Ultrasound_stillgan --norm instance -- batch_size 8 --gpu_ids 6 --no_dropout -- postname last --netG unet_combine_2layer
```

For Unpaired Structure Persevere Medical Image Enhancement:

```
python test_stillgan.py --dataroot ./datasets/ultrasound --name train_ultrasound_stillgan_twolow --model still_gan --input_nc 1 --output_nc 1 --direction AtoB --dataset_mode Ultrasound_stillgan --norm instance -- batch_size 8 --gpu_ids 6 --no_dropout -- postname last --netG resunet
```

Released soon.


## Trained model's weight

**Note:** If you want to use TAME in your own dataset, please re-train a new model with your own data, because it is a method based on domain adaptation, which means it needs target data (without ground truth) in the training phase.

For the model of TAME 'Enhancing and Adapting in the Clinic: Test-time Adaptation for Medical Image Enhancement' please download the pretrained model and place the document based on the following table:

|        | Baidu Cloud                                                  | Google Cloud                                                 | Directory                                        |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| TAME | Coming soon                                                  | Coming soon                                                  | project_root/checkpoints/TAME/latest_net_G.pth |


# Citation

```
@inproceedings{li2022structure,
  title={Structure-consistent restoration network for cataract fundus image enhancement},
  author={Li, Heng and Liu, Haofeng and Fu, Huazhu and Shu, Hai and Zhao, Yitian and Luo, Xiaoling and Hu, Yan and Liu, Jiang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={487--496},
  year={2022},
  organization={Springer}
}
```
