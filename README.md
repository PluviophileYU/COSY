# COSY
This repository contains the code for the following paper:
* Sicheng Yu, Hao Zhang, Yulei Niu, Qianru Sun, Jing Jiang *"COSY: COunterfactual SYntax for Cross-Lingual Understanding (https://aclanthology.org/2021.acl-long.48/)

## Requirement
* torch 1.2.0
* transformers 3.0.2
* apex 0.1
* tensorboardX 1.8
* prettytable 0.7.2

## Cross-lingual Question Answering (MLQA and XQuad)

### Dowload data
- Step 1: Download training dataset SQuAD via this link (https://drive.google.com/file/d/10STENBjVF3XaCKvzBudtC32B4RWzOwyg/view?usp=sharing) (we rename the file name), and store them in directory `/XQA/data/train`.
- Step 2: Download our extracted syntax feature via this link (https://drive.google.com/drive/folders/12Z6LcjUSYzJACMjIMUJ19DObVccVLEQu?usp=sharing), and store them in directory `/XQA/data/cache`. These files is a bit large, please reserve about 10GB for it.

### Training and Inference
```sh
cd XQA/src
bash run.sh
```

### COSY cross-lingual question answering model trained by me
You can download the model trained by us (with mBERT). You can find the results we reported in our paper. () 

