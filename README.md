# Not End-to-End: Explore Multi-Stage Architecture for Online Surgical Phase Recognition

## Accepted by ACCV2022

## Enviroment
Pytorch == 1.1.0, torchvision == 0.3.0, python == 3.6 CUDA=10.0

## Reproduce our results
Download the dataset and the pretrained models at https://pan.baidu.com/s/1I9WrHH0KKeebsRWtGK4fdA to reproduce our results.

```
python main.py --action=base_predict
python main.py --action=refine_predict
```

Evaluation code is obtained in the dataset. (cholec80/eva/Main.m).
You will get Acc=88.7% and 92.6%, respectively.

## Train the model with your own settings
```
python main.py --action=base_train
python main.py --action=refine_train
```


## [Optical]
We also provide the code for generating disturbed prediction sequence, but it is not necessary because we have provided the generated sequence in the download dataset.
```
python main.py --action=cross_validate_type
python main.py --action=mask_hard_frame_type
python main.py --action=random_mask_type
```

We also provide the code for extract inceptionv3 features for video frames and find out hard frames in the dataset. Again, this is not a necessary step.   
```
python frame_feature_extractor.py --action=train --dataset=cholec80
python frame_feature_extractor.py --action=extract --dataset=cholec80 --target=train_set/test_set
python frame_feature_extractor.py --action=hard_frame --dataset=cholec80 --target=train_set --k=0/1/2.../9
python frame_feature_extractor.py --action=hard_frame --dataset=cholec80 --target=test_set
```
