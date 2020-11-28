# Final-Project-Group8

### Getting Started

$ mkdir Project

$ cd Project

$ mkdir Data

$ git clone git@github.com:datsgwu/Final-Project-Group8.git

$ git clone https://github.com/fizyr/keras-retinanet

$ cd keras-retinanet

$ python3 setup.py build_ext --inplace

$ pip install progressbar2

### After running annotate.py and get_model.py

$ cd Project/keras-retinanet/

$ python3 keras_retinanet/bin/train.py --freeze-backbone --weights 'snapshots/resnet50_coco_best_v2.1.0.h5' --batch-size 8 --steps 2500 --epochs 15 csv '/Project/Data/images/train.csv' '/Project/Data/images/classes.csv'

The above ^^^ needs some reconfiguring but it should at least run through 30 steps, otherwise there is an issue


### Resources

https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5

Data -- https://captain-whu.github.io/DOAI2019/dataset.html

Data -- https://drive.google.com/drive/folders/1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK
