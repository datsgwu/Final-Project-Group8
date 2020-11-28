# Final-Project-Group8

### Getting Started

$ pip install pycocotools

$ pip install scikit-image

$ pip install gdown

$ cd /home/ubuntu

$ mkdir data

$ cd data

$ gdown https://drive.google.com/uc?id=1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2

$ unzip -u part1.zip

$ mkdir annotations

$ gdown https://drive.google.com/uc?id=12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX

$ unzip DOTA-v1.5_train.zip -d annotations

$ cd ..

$ mkdir project

$ cd project

$ git clone git@github.com:datsgwu/Final-Project-Group8.git

$ git clone https://github.com/yhenon/pytorch-retinanet.git


### After running annotate.py and get_model.py...

$ cd /home/ubuntu/project/pytorch-retinanet/

$ python3 train.py --dataset csv --csv_train /home/ubuntu/data/train.csv --csv_classes /home/ubuntu/data/classes.csv --csv_val /home/ubuntu/data/test.csv --epochs 200


### Evaluate with evaluate.py

