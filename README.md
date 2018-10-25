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
├── model_zoo
│   ├── model_info.txt
│   └── trained_model
├── models
│   ├── blockunits.py
│   ├── drcn.py
│   ├── espcn.py
│   ├── fsrnn.py
│   ├── rednet.py
│   ├── srcnn.py
│   ├── srdensenet.py
│   └── vdsr.py
├── utils
│   ├── extractor.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── runner.py
│   └── vision.py
...
```
#### directories
* `./src/data_dir`: original images
* `./dataset/save_dir`: croped images and related information
* `./model_zoo`: pretrained and trained models with related information
* `./models`: model architecture
* `./utils`: utilities

#### scripts
* `extractor.py`: extract crops from big images saved in `./data/data_dir` with different methods, save crops and related information in `./dataset/save_dir`
* `preprocess.py`: data augmentation
* `loader.py`: load images from `./data/data_dir` with data augmentation
* `metrics.py`: evaluation metrics such as PSNR
* `runner.py`: training, testing, log saving 

#### files
* `./dataset/save_dir/all.csv trian.csv test.csv val.csv`: image names(id)
* `./dataset/save_dir/statistic.csv`: the way of obtaining data
* `./model_zoo/model_info.txt`: model argument information, more information can be found in dir `./logs/statistic`

## Model Architecture
[Here](https://gitlab.com/Chokurei/geosr/tree/master/models)

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
