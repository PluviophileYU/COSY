# COSY
This repository contains the code for the following paper:
* Sicheng Yu, Hao Zhang, Yulei Niu, Qianru Sun, Jing Jiang "COSY: COunterfactual SYntax for Cross-Lingual Understanding" (https://aclanthology.org/2021.acl-long.48/)

## Requirement
* torch 1.2.0
* transformers 3.0.2
* apex 0.1
* tensorboardX 1.8
* prettytable 0.7.2

## Cross-lingual Question Answering (MLQA and XQuAD)

### Dowload data
- Step 1: Download training dataset SQuAD via this link (https://drive.google.com/file/d/10STENBjVF3XaCKvzBudtC32B4RWzOwyg/view?usp=sharing) (we rename the file name), and store them in directory `/XQA/data/train`.
- Step 2: Download our extracted syntax feature via this link (https://drive.google.com/drive/folders/12Z6LcjUSYzJACMjIMUJ19DObVccVLEQu?usp=sharing), and store them in directory `/XQA/data/cache`. These files is a bit large, please reserve about 20GB for it.

### Training and Inference
```sh
cd XQA/src
bash run.sh
```

### QA model trained by me
You can download the model trained by us (with mBERT). You can find the results we reported in our paper. (https://drive.google.com/drive/folders/1SR1mnssCugo81LuucykFixl00LmO347T?usp=sharing) 

Recently I am quite busy with the internship and deadline thus the code is still a bit messy. I will come back for the refinement later.

# Citation
If you feel this project helpful to your research, please cite our work.
```
@inproceedings{yu-etal-2021-cosy,
    title = "{COSY}: {CO}unterfactual {SY}ntax for Cross-Lingual Understanding",
    author = "Yu, Sicheng  and
      Zhang, Hao  and
      Niu, Yulei  and
      Sun, Qianru  and
      Jiang, Jing",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.48",
    doi = "10.18653/v1/2021.acl-long.48",
    pages = "577--589",
}
```
