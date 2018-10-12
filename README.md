# Geosr - A Computer Vision Package for Remote Sensing Image Super Resolution
High Resolution: PSNR            |  Low Resolution: 33.44 |  Super Resolution: 36.03
:-------------------------:|:-------------------------:|:-------------------------:
![hr_img2](/uploads/78c541a647afdb8820cfa0b682a96820/hr_img2.png)  |  ![lr_img2](/uploads/ddb75a8f6e9c7e498c89ac02deeb69e0/lr_img2.png)  |  ![sr_img2](/uploads/55831ea1829bcb47b65d6911ef60d783/sr_img2.png)

## Structure of directories
### sub directories
```
Geosr
├── data
│   └── image.png
├── models
│   ├── blockunits.py
│   ├── drcn.py
│   ├── espcn.py
│   ├── fsrnn.py
│   ├── rednet.py
│   ├── srcnn.py
│   ├── srdensenet.py
│   └── vdsr.py
├── utils
│   ├── datasets.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── preprocess.py
│   ├── runner.py
│   └── vision.py
...
```

## Model Architecture
[Here](https://gitlab.com/Chokurei/geosr/tree/master/models)
