# DisT

This is the pytorch implementation of our paper: **Discriminative Training Framework**.

## Environment
- python 3.11.5
- pytorch 2.0.1
- numpy 1.26.3 


## Commands

We provided the clustered users, spec_users_{dataset}.pth, for all datasets in the [data](https://github.com/hyNTU/DisT/tree/main/data) folder:
### To obtain contrastive loss 
Go to NCF or CDAE folder and run the codes below:
#### 1) Overall Loss
```
python storage.py --dataset movielens --epoch_eval 0 --gpu 0
```
or
```
python storage_CDAE.py --dataset movielens --epoch_eval 0 --gpu 0
```
#### 2) Group-specific Loss
```
python storage_user_specific.py --dataset movielens --epoch_eval 0 --gpu 0 --model_num 5
```
or
```
python storage_user_specific_CDAE.py --dataset movielens --epoch_eval 0 --gpu 0 --model_num 5
```
The two codes above will store the losses which will be used for denoising asymmetric instances below.

### To run GMF
Go to NCF folder and simply run the code below with default settings to return results shown in the paper:
```
python main.py --dataset movielens --model GMF --gpu 0 --remove 0.5 --epoch_eval 20

```
or for NeuMF
```
python main.py --dataset yelp --model NeuMF-end --gpu 0 --remove 0.5 --epoch_eval 20
```
### To run CDAE
Go to CDAE folder and run the code below:
```
python main_CDAE.py --dataset movielens --gpu 0 --remove 0.5 --epoch_eval 20

```

To change the hyperparameter settings, modify `--remove` & `--model_num` which controls r & K in the paper respectively.


## Citation  
If you use our code, please kindly cite:

```

```
