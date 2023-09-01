# Structure-consistent Restoration Network for Cataract Fundus Image Enhancement (for the first phase)

We propose a method of structure-consistent restoration network for cataract fundus image enhancement [[arXiv]](https://arxiv.org/abs/2206.04684). 

![](./images/scrnet_overview.png)


# Unpaired Structure Persevere Medical Image Enhancement (for the second phase)

We propose a method of unpaired structure persevere network for medical image enhancement. 

![](./images/scrnet_overview.png)



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

For ArcNet:

```
python train.py --dataroot ./images/cataract_dataset --name arcnet --model arcnet --netG unet_256 --input_nc 6 --direction AtoB --dataset_mode cataract_guide_padding --norm batch --batch_size 8 --gpu_ids 0
```

For SCR-Net:

```
python train.py --dataroot ./images/cataract_dataset --name scrnet --model scrnet --input_nc 3 --direction AtoB --dataset_mode cataract_with_mask --norm instance --batch_size 8 --gpu_ids 0 --lr_policy linear --n_epochs 150 --n_epochs_decay 50
```


Released soon.

## Test & Visualization

For ArcNet:

```
python test.py --dataroot ./images/cataract_dataset --name arcnet --model arcnet --netG unet_256 --input_nc 6 --direction AtoB --dataset_mode cataract_guide_padding --norm batch --gpu_ids 0
```

For ScrNet:

```
python test.py --dataroot ./images/cataract_dataset --name scrnet --model scrnet --netG unet_combine_2layer --direction AtoB --dataset_mode cataract_with_mask --input_nc 3 --output_nc 3
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
