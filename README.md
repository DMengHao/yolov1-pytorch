# yolov1-pytorch
This is the YOLOv1 version implemented using Pytorch.
## 🌐 Usage
### 🏊 Training
**1. Virtual Environment**
Just use the official YOLOv5 version of the environment, the link is [this link](https://github.com/ultralytics/yolov5)
```
# create virtual environment
conda create -n cddfuse python=3.8.10
conda activate cddfuse
pip install -r requirements.txt
```

**2. Data Preparation**

Download the VOC dataset from [this link](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and place it in the folder ``'./'``.

**3. Pre-Processing**

Run 
```
python write_txt.py
``` 
and the processed train and val dataset is in ``'./'``.

**4. YOLOv1 Training**

Run 
```
python train.py
# DP训练
python DP_train.py
# DDP训练
pyhton DDP_train.py
``` 

### 🏄 Testing

Run 
```
python predict.py 
``` 



