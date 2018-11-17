# Geosr - A Computer Vision Package for Remote Sensing Image Super Resolution
| Type         | Low Resolution                                                                                                | High Resolution                                                                                        | BICUBIC Interpolation                                                                                                    | Super Resolution                                                                                                                                            |
|--------------|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| church       | ![church_194_lr_up2](/uploads/74b679b6a952cf29f1930b11e27f75e9/church_194_lr_up2.jpg)             | ![church_194](/uploads/7de30114c9e567fb7a27a5ab36bfe194/church_194.jpg)                   | ![church_194_lr_up2_lerp](/uploads/b5523b946b1a9b996d0f12bd285ac437/church_194_lr_up2_lerp.jpg)             | ![church_194_up2_ESPCN_epoch_100_Nov17_10](/uploads/dcd16449905f1ac9b87392e41c08ddc2/church_194_up2_ESPCN_epoch_100_Nov17_10.jpg)             |
| airplane     | ![airplane_624_lr_up2](/uploads/20a08a70976bd084fe56f6fe8c392b17/airplane_624_lr_up2.jpg)         | ![airplane_624_lr_up1](/uploads/edf479b1991065c24b063b49d0283d66/airplane_624_lr_up1.jpg) | ![airplane_624_lr_up2_lerp](/uploads/ab63a81b94516a09b821c71a17246b54/airplane_624_lr_up2_lerp.jpg)         | ![airplane_624_up2_ESPCN_epoch_100_Nov17_00](/uploads/0adb676a92f4e9f7d471f35b2efe60e6/airplane_624_up2_ESPCN_epoch_100_Nov17_00.jpg)         |
| river        | ![river_085_lr_up2](/uploads/50fe7676ed64af97368ba6a33add99e3/river_085_lr_up2.jpg)               | ![river_085](/uploads/4edff25eda3eb94e838650349ecd75a1/river_085.jpg)                     | ![river_085_lr_up2_lerp](/uploads/948a2ab64220282b8d1b459fc9aa0b7b/river_085_lr_up2_lerp.jpg)               | ![river_085_up2_ESPCN_epoch_100_Nov17_10](/uploads/2c65e2da42413141458ff88322f7cbfe/river_085_up2_ESPCN_epoch_100_Nov17_10.jpg)               |
| runway       |![runway_389_lr_up2](/uploads/3cad2874cc185eb9fe75b3e369290564/runway_389_lr_up2.jpg)    | ![runway_389](/uploads/3a27c51dbc66adb79b53a666be4d5873/runway_389.jpg)                   |     ![runway_389_lr_up2_lerp](/uploads/261d66420f42ddbd7145d8e4b6a1d5b7/runway_389_lr_up2_lerp.jpg)                  | ![runway_389_up2_ESPCN_epoch_100_Nov17_10](/uploads/d38c4438d0377bf1027dbd8cd178d8e6/runway_389_up2_ESPCN_epoch_100_Nov17_10.jpg)             |
| ship         | ![ship_295_lr_up2](/uploads/5eeb4ee1ab33422c9d850cac1b68da5f/ship_295_lr_up2.jpg)                 | ![ship_295](/uploads/b90b44339895c8969fcbfa29ab29c873/ship_295.jpg)                       | ![ship_295_lr_up2_lerp](/uploads/7b28b069a84b543564617f664333be06/ship_295_lr_up2_lerp.jpg)                 | ![ship_295_up2_ESPCN_epoch_100_Nov17_10](/uploads/f64fddf16b7e738b1abbd4ec3a1d347f/ship_295_up2_ESPCN_epoch_100_Nov17_10.jpg)                 |
| tennis court | ![tennis_court_418_lr_up2](/uploads/95da0525ca8e3310163d27368608cdc2/tennis_court_418_lr_up2.jpg) | ![tennis_court_418](/uploads/f0c661c99495b1b4756690256aa0b100/tennis_court_418.jpg)       | ![tennis_court_418_lr_up2_lerp](/uploads/a8449e5d8792fdd054bfb93eeaddf774/tennis_court_418_lr_up2_lerp.jpg) | ![tennis_court_418_up2_ESPCN_epoch_100_Nov17_10](/uploads/dffacaf11cd7edd2cebb49517ee8ac37/tennis_court_418_up2_ESPCN_epoch_100_Nov17_10.jpg) |
## Structure of directories
### sub directories

<details><summary> Click Me </summary>
<p>

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
</p>
</details>

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
### Training
__Train or Not?__
```python
parser.add_argument('--train', type=lambda x: (str(x).lower() == 'true'), default=True, help='train or not?')
```

__Data Directory__
```python
parser.add_argument('--data_dir', type=str, default=os.path.join(DIR, 'dataset','church-alloc'), help="data directory for training")
```

__Hyperparameters__
```python
parser.add_argument('--crop_size', type=int, default=224, help='crop size from each data. Default=224 (same to image size)')
parser.add_argument('--nb_channel', type=int, default=1, help="input image band, based on band_mode")
parser.add_argument('--interpolation', type=lambda x: (str(x).lower() == 'true'), default=False, help="conduct pre-interpolation or not")
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--aug', type=lambda x: (str(x).lower() == 'true'), default=True, help='data augmentation or not') 
parser.add_argument('--aug_mode', type=str, default='c', choices=['a', 'b', 'c', 'd', 'e'], 
                    help='data augmentation mode: a, b, c, d, e')
parser.add_argument('--base_kernel', type=int, default=64, help="base kernel")
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--valbatch_size', type=int, default=10, help='validation batch size')
parser.add_argument('--evalbatch_size', type=int, default=10, help='evaluation batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                    help='trigger type for logging')
parser.add_argument('--interval', type=int, default=50,
                    help='interval for logging')
parser.add_argument('--middle_checkpoint', type=lambda x: (str(x).lower() == 'true'), default=False, help='save middle checkpoint of model or not')

parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'), default=True, help='use cuda?')    
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
```
### Testing
__Testing or Not?__
```python
parser.add_argument('--test', type=lambda x: (str(x).lower() == 'true'), default=True, help='test or not?')
```

__Testing Mode__
```python
parser.add_argument('--test_dir', type=str, default=os.path.join(DIR, 'dataset','church-alloc','images', 'test'), help="testing data directory")
parser.add_argument('--test_model_name', type=str, default='up8_ESPCN_epoch_100_Nov17_00.pth', help='model used for testing')

parser.add_argument('--ground_truth', type=lambda x: (str(x).lower() == 'true'), default=True, help='have ground truth or not')
parser.add_argument('--test_model', type=lambda x: (str(x).lower() == 'true'), default=True, help='test different models')
parser.add_argument('--test_middle_checkpoint', type=lambda x: (str(x).lower() == 'true'), default=False, help='have middle checkpoints of one model')
 
```

## Logs
### Learning Curve
| ![up2_ESPCN_epoch_100_Nov17_00](/uploads/218a161ea230c22001db99c3e4e52232/up2_ESPCN_epoch_100_Nov17_00.png) | ![up2_SRDenseNet_epoch_100_Nov17_01](/uploads/6423f84d89c1d8cce26fc4902d686f90/up2_SRDenseNet_epoch_100_Nov17_01.png) |
|:-----------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
### Model Info
#### model settings
|  aug | aug_mode | band_mode | base_kernel | batch_size | crop_size | cuda | data_dir                                      | date     | epochs | evalbatch_size | ground_truth | interpolation | interval | iter_interval | iters | lr   | method     | middle_checkpoint | model_name                            | nEpochs | nb_channel | save_model_epoch | seed | test  | test_diff_model | test_dir                                                  | test_middle_checkpoint | test_model | test_model_name                  | testbatch_size | threads | train | trigger | upscale_factor | valbatch_size |
|:----:|:--------:|-----------|-------------|------------|-----------|------|-----------------------------------------------|----------|--------|----------------|--------------|---------------|----------|---------------|-------|------|------------|-------------------|---------------------------------------|---------|------------|------------------|------|-------|-----------------|-----------------------------------------------------------|------------------------|------------|----------------------------------|----------------|---------|-------|---------|----------------|---------------|
| True | c        | Y         | 64          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_00 | 100    | 10             | True         | False         | 1        | 6             | 600   | 0.01 | ESPCN      | False             | up4_ESPCN_epoch_100_Nov17_00.pth      | 100     | 1          |                  | 123  | True  |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | False                  | True       | up2_ESPCN_epoch_100_Nov17_00.pth |                | 6       | True  | epoch   | 4              | 10            |
| True | c        | Y         | 64          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_00 | 100    | 10             | True         | False         | 1        | 6             | 600   | 0.01 | ESPCN      | False             | up8_ESPCN_epoch_100_Nov17_00.pth      | 100     | 1          |                  | 123  | True  |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | False                  | True       | up4_ESPCN_epoch_100_Nov17_00.pth |                | 6       | True  | epoch   | 8              | 10            |
| True | c        | Y         | 64          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_00 | 100    | 10             | False        | False         | 1        | 6             | 600   | 0.01 | FSRCNN     | False             | up2_FSRCNN_epoch_100_Nov17_00.pth     | 100     | 1          |                  | 123  | True  |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | False                  | True       | up2_ESPCN_epoch_200_Nov15_10.pth |                | 6       | True  | epoch   | 2              | 10            |
| True | c        | Y         | 16          | 64         | 224       | True | /media/kaku/Work/geosr/dataset/airplane-alloc | Nov17_01 | 100    | 10             | False        | False         | 1        | 6             | 600   | 0.01 | SRDenseNet | False             | up2_SRDenseNet_epoch_100_Nov17_01.pth | 100     | 1          |                  | 123  | False |                 | /media/kaku/Work/geosr/dataset/airplane-alloc/images/test | True                   | True       | up2_ESPCN_epoch_200_Nov15_10.pth |                | 6       | True  | epoch   | 2              | 10            |
#### performance
__train__

| date     | epochs | fps     | iters | method     | nb_samples | nrmse | psnr    | ssim  | time(sec) |
|----------|--------|---------|-------|------------|------------|-------|---------|-------|-----------|
| Nov16_22 | 100    | 639.04  | 600   | FSRCNN     | 420        | 0.057 | 23.538  | 0.905 | 0.657     |
| Nov16_22 | 100    | 627.644 | 600   | ESPCN      | 420        | 0.027 | 28.436  | 0.983 | 0.669     |
| Nov16_23 | 100    | 484.939 | 600   | SRCNN      | 420        | 0.024 | 31.288  | 0.993 | 0.866     |
| Nov16_23 | 100    | 235.047 | 600   | VDSR       | 420        | 0     | 118.102 | 1     | 1.787     |
| Nov16_23 | 100    | 309.75  | 600   | SRDenseNet | 420        | 0.034 | -35.837 |       | 1.356     |

<br>

__val__

| date     | epochs | fps     | iters | method     | nb_samples | nrmse | psnr    | ssim  | time(sec) |
|----------|--------|---------|-------|------------|------------|-------|---------|-------|-----------|
| Nov16_22 | 100    | 592.429 | 600   | FSRCNN     | 140        | 0.055 | 23.864  | 0.911 | 0.236     |
| Nov16_22 | 100    | 614.134 | 600   | ESPCN      | 140        | 0.026 | 28.946  | 0.985 | 0.228     |
| Nov16_23 | 100    | 460.971 | 600   | SRCNN      | 140        | 0.022 | 31.982  | 0.994 | 0.304     |
| Nov16_23 | 100    | 223.447 | 600   | VDSR       | 140        | 0     | 118.456 | 1     | 0.627     |
| Nov16_23 | 100    | 287.71  | 600   | SRDenseNet | 140        | 0.035 | -35.931 | 0.01  | 0.487     |

<br>

__test__

| date     | epochs | fps     | iters | method     | nb_samples | nrmse | psnr    | ssim  | time(sec) |
|----------|--------|---------|-------|------------|------------|-------|---------|-------|-----------|
| Nov16_19 | 200    | 427.941 | 400   | FSRCNN     | 50         | 0.115 | 19.083  | 0.869 | 0.117     |
| Nov16_19 | 200    | 336.643 | 400   | ESPCN      | 50         | 0.05  | 22.838  | 0.977 | 0.149     |
| Nov16_19 | 200    | 67.637  | 400   | SRCNN      | 50         | 0.027 | 29.993  | 0.995 | 0.739     |
| Nov16_19 | 200    | 87.139  | 400   | VDSR       | 50         | 0     | 119.635 | 1     | 0.574     |
| Nov16_19 | 200    | 241.894 | 400   | SRDenseNet | 50         | 0.035 | -39.654 |       | 0.207     |

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
