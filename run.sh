echo 'With Ground Truth'
python ./FSRCNN.py --nEpochs 200 --interval 50 --ground_truth True
python ./ESPCN.py --nEpochs 200 --interval 50 --ground_truth True
python ./SRCNN.py --nEpochs 200 --interval 50 --ground_truth True
python ./VDSR.py --nEpochs 200 --interval 50 --ground_truth True
python ./SRDenseNet.py --nEpochs 200 --interval 50 --ground_truth True
echo 'Without Ground Truth'
python ./FSRCNN.py --nEpochs 200 --interval 50 --ground_truth False
python ./ESPCN.py --nEpochs 200 --interval 50 --ground_truth False
python ./SRCNN.py --nEpochs 200 --interval 50 --ground_truth False
python ./VDSR.py --nEpochs 200 --interval 50 --ground_truth False
python ./SRDenseNet.py --nEpochs 200 --interval 50 --ground_truth False
echo 'End'


