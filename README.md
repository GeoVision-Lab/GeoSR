# Geosr - A Computer Vision Package for Remote Sensing Image Super Resolution
High Resolution: PSNR            |  Low Resolution: 33.32 |  Super Resolution: 36.26
:-------------------------:|:-------------------------:|:-------------------------:
![hr_img2](/uploads/78c541a647afdb8820cfa0b682a96820/hr_img2.png)  |  ![lr_img2](/uploads/ddb75a8f6e9c7e498c89ac02deeb69e0/lr_img2.png)  |  ![sr_img2](/uploads/55831ea1829bcb47b65d6911ef60d783/sr_img2.png)

## Structure of directories
### sub directories
```
Geosr
├── data
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
├── models
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
│   ├── datasets.py
│   ├── extractor.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── runner.py
│   └── vision.py
...
```
#### directories
* `./data/data_dir`: original images
* `./dataset/save_dir`: croped images
* ./models: model architecture

#### scripts
* extractor.py: Extract crops from big images saved in './data/data_dir' with different methods, save crops and related information in './dataset/save_dir'

## Model Architecture
[Here](https://gitlab.com/Chokurei/geosr/tree/master/models)

## Get Started
### Reference
If you use our code or any of the ideas from our paper please cite:
```
@article{wu2018geoseg,
  title={Geoseg: A Computer Vision Package for Automatic Building Segmentation and Outline Extraction},
  author={Wu, Guangming and Guo, Zhiling},
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
