mkdir -p bin
mkdir -p output
python3 ./ModelInference.py\
 -r 'facebookresearch/detr:main'\
 -m 'detr_resnet50'\
 -b 'bin'\
 -i './images/horse.jpg'\
 -v