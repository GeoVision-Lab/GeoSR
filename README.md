# Geosr - A Computer Vision Package for Remote Sensing Image Super Resolution
High Resolution: PSNR            |  Low Resolution: 33.32 |  Super Resolution: 36.26
:-------------------------:|:-------------------------:|:-------------------------:
![hr_img2](/uploads/78c541a647afdb8820cfa0b682a96820/hr_img2.png)  |  ![lr_img2](/uploads/ddb75a8f6e9c7e498c89ac02deeb69e0/lr_img2.png)  |  ![sr_img2](/uploads/55831ea1829bcb47b65d6911ef60d783/sr_img2.png)

## Structure of directories
### sub directories
```
Geosr
├── src
│   └── data_dir
├── dataset
│   └── save_dir
│       ├── all.csv
│       ├── train.csv
│       ├── test.csv
│       ├── val.csv
│       ├── statistic.csv
│       └── image
│           ├── train
│           ├── test
│           └── val
├── logs
│   ├── curve
│   ├── raw
│   └── statistic
│       ├── model_info.csv
│       ├── train.csv
│       ├── test.csv
│       └── val.csv
├── model_zoo
│   └── trained_model
├── archs
│   ├── blockunits.py
│   ├── drcn.py
│   ├── espcn.py
│   ├── fsrnn.py
│   ├── rednet.py
│   ├── srcnn.py
│   ├── srdensenet.py
│   └── vdsr.py
├── utils
│   ├── combiner.py
│   ├── extractor.py
│   ├── loader.py
│   ├── metrics.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── tester.py
│   └── vision.py
├── result
│   ├── raw
│   │   └── test_dir_name
│   │       ├── diff_model
│   │       │   ├── with_truth
│   │       │   │   ├── results(sr)
│   │       │   │   ├── result_avg_log.csv
│   │       │   │   └── result_log.csv
│   │       │   └── without_truth
│   │       │       └── results(sr)
│   │       └── middle_checkpoint
│   │           └── model_name
│   │               ├── with_truth
│   │               │   ├── results(sr)
│   │               │   ├── result_avg_log.csv
│   │               │   └── result_log.csv
│   │               └── without_truth
│   │                   └── results(sr)
│   └── generate
│       ├── combined
│       │   └── test_dir_name
│       │       └── results(sr)
│       ├── figures
│       └── tables
├── main.py
│  
...
```
#### directories
* `./src/data_dir`: original images
* `./dataset/save_dir`: croped images and related information
* `./model_zoo`: pretrained and trained models with related information
* `./archs`: model architectures
* `./utils`: utilities

#### scripts
* `extractor.py`: extract crops or from big images or randomly allocate original images saved in `./src/data_dir` with different methods, save crops and related information in `./dataset/save_dir`
* `preprocess.py`: data augmentation
* `loader.py`: load images from `./data/data_dir` with data augmentation
* `metrics.py`: evaluation metrics such as PSNR, SSIM, NRMSE, VIFP
* `trainer.py`: training, evaluation, log saving
* `tester.py`: testing, log saving
* 

#### files
* `./dataset/save_dir/all.csv trian.csv test.csv val.csv`: image names(id)
* `./dataset/save_dir/statistic.csv`: the way of obtaining data
* `./logs/statistic/model_info.csv`: model argument information
* `./logs/statistic/trian.csv test.csv val.csv` final statistic result and parameter information for related model

## Model Architectures
[Here](https://gitlab.com/Chokurei/geosr/tree/master/archs)

## Get Started
### Reference
If you use our code or any of the ideas from our paper please cite:
```
@article{wu2018geoseg,
  title={Geosr: A Computer Vision Package for Remote Sensing Image Super Resolution},
  author={Guo, Zhiling and Wu, Guangming},
  journal={arXiv preprint arXiv:1809.03175},
  year={2018}
}
```
### Requirements
* [Python 3.5.2+](https://www.python.org/)
* [torch 0.4.1+](https://pytorch.org/tutorials/)
* [torchvision 0.2.1+](https://pytorch.org/docs/stable/torchvision/index.html)

### Data
```
$ python ./utils/extractor.py --data_dir DATA_DIR --mode 'slide-rand'
```
or  
save existing training, valiation, and testing dataset in `./dataset/save_dir` respectively

### Preprocessing
__Band choose__  
```python
parser.add_argument('--band_mode', type=str, default='Y', choices=['Y', 'YCbCr', 'RGB'], help="band mode")
```
__Data augmentation__  
Choose data augmentation method in `./main.py`, detailed information in `./utils/preprocessor.py`
```python
parser.add_argument('--aug', type=lambda x: (str(x).lower() == 'true'), default=True, help='data augmentation or not')
parser.add_argument('--aug_mode', type=str, default='c', choices=['a', 'b', 'c', 'd', 'e'],
                        help='data augmentation mode: a, b, c, d, e')
```
## Logs
### Learning Curve
| ![up2_ESPCN_epoch_100_Nov17_00](/uploads/218a161ea230c22001db99c3e4e52232/up2_ESPCN_epoch_100_Nov17_00.png) | ![up2_SRDenseNet_epoch_100_Nov17_01](/uploads/6423f84d89c1d8cce26fc4902d686f90/up2_SRDenseNet_epoch_100_Nov17_01.png) |
|:-----------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
### Model Info
|  aug | aug_mode | band_mode | base_kernel | batch_size | crop_size | cuda | data_dir                                      | date     | epochs | evalbatch_size | ground_truth | interpolation | interval | iter_interval | iters | lr   | method     | middle_checkpoint | model_name                            | nEpochs | nb_channel | save_model_epoch | seed | test  | test_diff_model | test_dir                                                  | test_middle_checkpoint | test_model | test_model_name                  | testbatch_size | threads | train | trigger | upscale_factor | valbatch_size |
|:----:|:--------:|-----------|-------------|------------|-----------|------|-----------------------------------------------|----------|--------|----------------|--------------|---------------|----------|---------------|-------|------|------------|-------------------|---------------------------------------|---------|------------|------------------|------|-------|-----------------|-----------------------------------------------------------|------------------------|------------|----------------------------------|----------------|---------|-------|---------|----------------|---------------|
| True | c        | Y         | 64          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_00 | 100    | 10             | True         | False         | 1        | 6             | 600   | 0.01 | ESPCN      | False             | up4_ESPCN_epoch_100_Nov17_00.pth      | 100     | 1          |                  | 123  | True  |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | False                  | True       | up2_ESPCN_epoch_100_Nov17_00.pth |                | 6       | True  | epoch   | 4              | 10            |
| True | c        | Y         | 64          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_00 | 100    | 10             | True         | False         | 1        | 6             | 600   | 0.01 | ESPCN      | False             | up8_ESPCN_epoch_100_Nov17_00.pth      | 100     | 1          |                  | 123  | True  |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | False                  | True       | up4_ESPCN_epoch_100_Nov17_00.pth |                | 6       | True  | epoch   | 8              | 10            |
| True | c        | Y         | 64          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_00 | 100    | 10             | False        | False         | 1        | 6             | 600   | 0.01 | FSRCNN     | False             | up2_FSRCNN_epoch_100_Nov17_00.pth     | 100     | 1          |                  | 123  | True  |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | False                  | True       | up2_ESPCN_epoch_200_Nov15_10.pth |                | 6       | True  | epoch   | 2              | 10            |
| True | c        | Y         | 16          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_01 | 100    | 10             | False        | False         | 1        | 6             | 600   | 0.01 | SRDenseNet | False             | up2_SRDenseNet_epoch_100_Nov17_01.pth | 100     | 1          |                  | 123  | False |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | True                   | True       | up2_ESPCN_epoch_200_Nov15_10.pth |                | 6       | True  | epoch   | 2              | 10            |
## Results
### Model Difference (diff_model)
#### table 
__result_avg_log.csv__  

| model                             | psnr_avg | ssim_avg | nrmse_avg |
|-----------------------------------|----------|----------|-----------|
| up2_ESPCN_epoch_100_Nov16_22      | 30.048   | 0.984    | 0.03      |
| up2_FSRCNN_epoch_100_Nov16_22     | 24.642   | 0.906    | 0.066     |
| up2_SRCNN_epoch_100_Nov16_23      | 33.034   | 0.994    | 0.023     |
| up2_SRDenseNet_epoch_100_Nov16_23 | -35.027  |          | 0.043     |
| up2_VDSR_epoch_100_Nov16_23       | 119.484  | 1        | 0         |
  
<br>  

__result_log.csv__  

| ip               | model                             | psnr    | ssim  | nrmse |
|------------------|-----------------------------------|---------|-------|-------|
| airplane_349.jpg | up2_SRCNN_epoch_100_Nov16_23      | 34.478  | 0.018 | 0.996 |
| airplane_349.jpg | up2_VDSR_epoch_100_Nov16_23       | 118.171 | 0     | 1     |
| airplane_349.jpg | up2_SRDenseNet_epoch_100_Nov16_23 | -36.341 | 0.053 | 0.008 |
| airplane_349.jpg | up2_ESPCN_epoch_100_Nov16_22      | 30.486  | 0.024 | 0.986 |
| airplane_349.jpg | up2_FSRCNN_epoch_100_Nov16_22     | 24.278  | 0.06  | 0.913 |
#### Visualization (results(sr))
| Type/PSNR | LR                                                                                                                                      | HR                                                                                                                                    | BICUBIC                                                                                                                             | ESPCN/30.486                                                                                                                                    |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Image     |                        ![airplane_349_lr_up2](/uploads/bdcd1aff7f57c41ee910e6d5c7c154a0/airplane_349_lr_up2.jpg)                        |                       ![airplane_349_lr_up1](/uploads/24e26a98afd1d9190a5922fbb5946b12/airplane_349_lr_up1.jpg)                       |                 ![airplane_349_lr_up2_lerp](/uploads/e3b7c4247c4679623f82e435a3c63d63/airplane_349_lr_up2_lerp.jpg)                 |      ![airplane_349_up2_ESPCN_epoch_100_Nov16_22](/uploads/347f1d656f773a01b677f3e83bd47b31/airplane_349_up2_ESPCN_epoch_100_Nov16_22.jpg)      |
| Type      | FSRCNN/24.278                                                                                                                           | SRCNN/34.478                                                                                                                          | VDSR/118.171                                                                                                                        | SRDenseNet/-36.431                                                                                                                              |
| Image     | ![airplane_349_up2_FSRCNN_epoch_100_Nov16_22](/uploads/bb6f56a1d889c3d71add00b9ff7455da/airplane_349_up2_FSRCNN_epoch_100_Nov16_22.jpg) | ![airplane_349_up2_SRCNN_epoch_100_Nov16_23](/uploads/9f8807936d31a3426fc7cada9b90bbbc/airplane_349_up2_SRCNN_epoch_100_Nov16_23.jpg) | ![airplane_349_up2_VDSR_epoch_100_Nov16_23](/uploads/4bdba5e42a236a81e7a4655a63273acb/airplane_349_up2_VDSR_epoch_100_Nov16_23.jpg) | ![airplane_349_up2_SRDenseNet_epoch_100_Nov16_23](/uploads/e89b6eb926c769b97c0d73b0378fc66d/airplane_349_up2_SRDenseNet_epoch_100_Nov16_23.jpg) |
### Epoch Difference (middle_checkpoint)
#### table
__result_avg_log.csv__ 

| model              | psnr_avg | ssim_avg | nrmse_avg |
|--------------------|----------|----------|-----------|
| epoch_10_iter_60   | 21.91    | 0.843    | 0.085     |
| epoch_20_iter_120  | 24.249   | 0.931    | 0.057     |
| epoch_30_iter_180  | 27.229   | 0.951    | 0.04      |
| epoch_40_iter_240  | 28.263   | 0.966    | 0.036     |
| epoch_50_iter_300  | 26.874   | 0.968    | 0.043     |
| epoch_60_iter_360  | 27.56    | 0.974    | 0.039     |
| epoch_70_iter_420  | 29.149   | 0.979    | 0.033     |
| epoch_80_iter_480  | 29.381   | 0.982    | 0.032     |
| epoch_90_iter_540  | 30.06    | 0.983    | 0.03      |
| epoch_100_iter_600 | 30.048   | 0.984    | 0.03      |

<br>  

__result_log.csv__ 

| ip               | model              | psnr   | ssim  | nrmse |
|------------------|--------------------|--------|-------|-------|
| airplane_349.jpg | epoch_10_iter_60   | 21.093 | 0.094 | 0.85  |
| airplane_349.jpg | epoch_20_iter_120  | 23.721 | 0.052 | 0.938 |
| airplane_349.jpg | epoch_30_iter_180  | 27.153 | 0.032 | 0.958 |
| airplane_349.jpg | epoch_40_iter_240  | 28.298 | 0.028 | 0.972 |
| airplane_349.jpg | epoch_50_iter_300  | 27.244 | 0.034 | 0.972 |
| airplane_349.jpg | epoch_60_iter_360  | 27.658 | 0.03  | 0.977 |
| airplane_349.jpg | epoch_70_iter_420  | 29.506 | 0.025 | 0.982 |
| airplane_349.jpg | epoch_80_iter_480  | 29.759 | 0.025 | 0.984 |
| airplane_349.jpg | epoch_90_iter_540  | 30.455 | 0.024 | 0.985 |
| airplane_349.jpg | epoch_100_iter_600 | 30.486 | 0.024 | 0.986 |

#### Visualization (results(sr))
| Epoch/PSNR | 10/21.093                                                                                                              | 20/23.721                                                                                                              | 30/27.153                                                                                                              | 40/28.298                                                                                                              | 50/27.244                                                                                                                |
|-------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Image |  ![airplane_349_epoch_10_iter_60](/uploads/2ea532877a7a64a80b1bc679d5d8e44c/airplane_349_epoch_10_iter_60.jpg)  | ![airplane_349_epoch_20_iter_120](/uploads/3b95ee27dee1483631bea27ce4a372d7/airplane_349_epoch_20_iter_120.jpg) | ![airplane_349_epoch_30_iter_180](/uploads/b43f5e3bd1c6f204acb68df2c4068c2b/airplane_349_epoch_30_iter_180.jpg) | ![airplane_349_epoch_40_iter_240](/uploads/26e7d22aab153832c4bd1d59655e788e/airplane_349_epoch_40_iter_240.jpg) | ![airplane_349_epoch_50_iter_300](/uploads/c72b6e6d0557791e9af38c8756276858/airplane_349_epoch_50_iter_300.jpg)   |
| Epoch/PSNR | 60/27.658                                                                                                              | 70/29.506                                                                                                              | 80/29.759                                                                                                              | 90/30.455                                                                                                              | 100/30.486                                                                                                               |
| Image | ![airplane_349_epoch_60_iter_360](/uploads/33e2f6dcd04ee3c91c6de54294142e5b/airplane_349_epoch_60_iter_360.jpg) | ![airplane_349_epoch_70_iter_420](/uploads/49f550f34f056cd73214fafe59b3ce7d/airplane_349_epoch_70_iter_420.jpg) | ![airplane_349_epoch_80_iter_480](/uploads/b7c8976853cd593a6b356c8a05b0efd5/airplane_349_epoch_80_iter_480.jpg) | ![airplane_349_epoch_90_iter_540](/uploads/1b7e34454491f862027487c835f9f9d6/airplane_349_epoch_90_iter_540.jpg) | ![airplane_349_epoch_100_iter_600](/uploads/852f878c7b8666504b69fc63dc629330/airplane_349_epoch_100_iter_600.jpg) |

### Upscale Difference
| Image | LR                                                                                        | BICUBIC                                                                                             | SR                                                                                                                                    |
|-------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Up_2  | ![airplane_349_lr_up2](/uploads/a522dfa54f361e233b8f144af8e73197/airplane_349_lr_up2.jpg) | ![airplane_349_lr_up2_lerp](/uploads/f0b040cb68e865d1948a2f668a4ff137/airplane_349_lr_up2_lerp.jpg) | ![airplane_349_up2_ESPCN_epoch_100_Nov17_00](/uploads/84d530d1ef1505362ddc335b61fa11cf/airplane_349_up2_ESPCN_epoch_100_Nov17_00.jpg) |
| Up_4  | ![airplane_349_lr_up4](/uploads/7ec46eda0c02f484d58ec3f01567f25e/airplane_349_lr_up4.jpg) | ![airplane_349_lr_up4_lerp](/uploads/cb0fcada26b257df780e7bc6b0cd5f8f/airplane_349_lr_up4_lerp.jpg) | ![airplane_349_up4_ESPCN_epoch_100_Nov17_00](/uploads/a78c8885509301c5f851d8b49393f4e1/airplane_349_up4_ESPCN_epoch_100_Nov17_00.jpg) |
| Up_8  | ![airplane_349_lr_up8](/uploads/0273a8b7677c74edae4df626a15fa855/airplane_349_lr_up8.jpg) | ![airplane_349_lr_up8_lerp](/uploads/5b9154d33a411934e1762ab3be4467a9/airplane_349_lr_up8_lerp.jpg) | ![airplane_349_up8_ESPCN_epoch_100_Nov17_00](/uploads/330b8f118bedaadf6db7902b2130f0c3/airplane_349_up8_ESPCN_epoch_100_Nov17_00.jpg) |
