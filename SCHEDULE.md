# Geosr Working Log
## 2018/10/9
1. /reference/SRCNN-keras/  
revise main.py, prepare_data.py  
result: rs_img(hr), rs_in(lr), rs_pre(sr)

## 2018/10/10
### ./models
1. espcn.py  
2. fsrcnn.py  
3. srcnn.py
### ./utils
1. vision.py: result comparison

## 2018/10/11
### ./models
1. blockunits.py
2. drcn.py
3. rednet.py
4. vdsr.py

## 2018/10/12
### ./models
1. srdensenet.py

## 2018/10/12 ~ 10/15
Finished entire geosr-keras version  
URL: https://gitlab.com/Chokurei/geosr_keras

## 2018/10/16
### .
1. ESPCN.py  
The original main.py, can be conducted
### ./utils
1. data.py
2. datasets.py

## 2018/10/17
### ./utils
extractor.py  
tow mode: stride/random

## 2018/10/18~19
### ./utils
preprocessor.py and loader.py
for loading data and data augmentation

## 2018/10/20
### ./utils
1. runner.py  
learn wu's code, still do not converse  
2. metrics.py  
new metrics need to be added

## 2018/10/22~27
### ./Utils
1. runner.py
can be conducted
2. ESPCN.py
optimized according to runner.py
