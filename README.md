# Learning Phase Transition Priors for Online Surgical Phase Recognition.

## Enviroment
Pytorch == 1.1.0, torchvision == 0.3.0, python == 3.7 CUDA=10.0

## Reproduce our results
Download the dataset at https://disk.pku.edu.cn:443/link/DE7DECE90BA70E601AB2E9DF1B6F2D45  Valid Until: 2023-04-03 23:59
Download the trained models at https://disk.pku.edu.cn:443/link/43DE87ACD388496FF6BAEB14D0D5391E Valid Until: 2023-04-03 23:59

```
python main.py --action=base_predict
python main.py --action=refine_predict
```
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
python frame_feature_extractor --action=train/extract
python frame_feature_extractor --action=hard_frame
```
