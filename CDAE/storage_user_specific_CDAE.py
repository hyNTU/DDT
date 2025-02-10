import os
import time
import argparse
import numpy as np
import random
import pandas as pd
import scipy.sparse as sp
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import copy

from tqdm import tqdm
import toolz
import model
import evaluate
import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'yelp')
parser.add_argument('--seed', 
	type = int,
	help = 'seed for reproducibility',
	default = 2019)
parser.add_argument("--gpu", 
	type=str,
	default="1",
	help="gpu card ID")
parser.add_argument("--epoch_eval", 
    type = int,
	default=10,
	help="epoch to start evaluation")
parser.add_argument("--batch_size", 
    type = int,
	default=2048,
	help="epoch to start evaluation")
parser.add_argument("--top_k", 
    type = list,
	default= [50, 100],
	help="compute metrics @k")
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 6,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--model_num', 
	type = int, 
	default = 5,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--exponent', 
	type = float, 
	default = 1, 
	help='exponent of the drop rate {0.5, 1, 2}')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(args.seed) # cpu
torch.cuda.manual_seed(args.seed) #gpu
np.random.seed(args.seed) #numpy
random.seed(args.seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


data_path = f'../data/{args.dataset}/'
model_path = f'../models/'



############################## PREPARE DATASET ##########################

train_data, valid_data, train_data_pos, valid_data_pos, test_data_pos, user_pos, user_num ,item_num, train_mat, valid_mat, train_data_noisy = data_utils.load_all(f'{args.dataset}', data_path)


train_mat_dense = train_mat.toarray()
users_list = np.array([i for i in range(user_num)])
train_dataset = data_utils.DenseMatrixUsers(users_list ,train_mat_dense)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

valid_mat_dense = valid_mat.toarray()
valid_dataset = data_utils.DenseMatrixUsers(users_list, valid_mat_dense)
valid_loader = data.DataLoader(valid_dataset, batch_size=4096, shuffle=True)



########################### LOSS FUNCTION #####################################
def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    loss_storage = loss_mul[t > 0.]
    assert loss_storage.shape == t[t>0.].shape
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update, loss_storage

def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate


########################### Eval #####################################
def eval(model, valid_loader, valid_data_pos, train_mat, best_recall, count, epoch_loss, map_dict_old, map_dict_new):
    top_k = args.top_k
    model.eval()
    # model prediction can be more efficient instead of looping through each user, do it by batch
    predictedIndices_all = torch.empty(len(valid_loader.dataset), top_k[-1], dtype=torch.long) # predictions
    GroundTruth = list(valid_data_pos.values()) # ground truth is exact item indices
    for user_valid, data_value_valid in valid_loader:
        with torch.no_grad():
            user_valid = user_valid.cuda()
            user_valid_sub = [map_dict_old[x.item()] for x in user_valid]
            prediction_input_from_train = torch.tensor(train_mat[user_valid_sub]).cuda()
            prediction = model(user_valid, prediction_input_from_train) # prediction of the batch from train matrix
            valid_data_mask = train_mat[user_valid_sub] * -9999# depends on the size of data

            prediction = prediction + torch.tensor(valid_data_mask).float().cuda()
            _, indices = torch.topk(prediction, top_k[-1])
            predictedIndices_all[user_valid.cpu()] = indices.cpu()


    user_valid_sub_new = [map_dict_new[x] for x in valid_data_pos.keys()]

    predictedIndices = predictedIndices_all[user_valid_sub_new]
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    print(f"Recall:{recall} NDCG: {NDCG}")
    if recall[0] > best_recall:
        best_recall = recall[0]
        count = 0
        torch.save(epoch_loss, f'../data/group_loss_CDAE_{args.dataset}.pth')

    else: 
        count += 1
    return best_recall, count
    



########################### BEGIN USER LOOP ###########################
list_of_users = torch.load(f'../data/spec_users_{args.dataset}.pth', weights_only=True)
print("LOADED USERS")
epoch_loss = torch.tensor(train_mat_dense).clone().cpu()


for i, group in enumerate(list_of_users):
    new_users_list = np.array([i for i in range(len(group))])
    map_dict_old = {i : key.item() for i,key in enumerate(group)}
    map_dict_new = {key.item(): i for i,key in enumerate(group)}
    train_loader.dataset.change_data(new_users_list, train_mat_dense[group.numpy()])
    valid_loader.dataset.change_data(new_users_list, valid_mat_dense[group.numpy()])
    valid_pos_filtered = {key.item(): valid_data_pos[key.item()] for key in group if key.item() in valid_data_pos}
    print(f"DATA FILTERED FOR GROUP {i}")

########################### CREATE MODEL #################################
    model_test = model.CDAE(len(group), item_num, 32, 0.2)
    model_test.cuda()
    BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
    num_ns = 1 # negative samples
    optimizer = optim.Adam(model_test.parameters(), lr=0.001)

########################### TRAINING #####################################

    top_k = args.top_k
    best_recall = 0
    count = 0 
    counter = 0


    for epoch in range(1000):
        model_test.train()
        train_loss = 0

        for user, data_value in train_loader:
            user = user.cuda()
            data_value = data_value.cuda()
            prediction = model_test(user, data_value)
            #negative sampling
            with torch.no_grad():
                num_ns_per_user = data_value.sum(1) * num_ns
                negative_samples = []
                users = []
                for u in range(data_value.size(0)):
                    batch_interaction = torch.randint(0, item_num, (int(num_ns_per_user[u].item()),))
                    negative_samples.append(batch_interaction)
                    users.extend([u] * int(num_ns_per_user[u].item()))


            negative_samples = torch.cat(negative_samples, 0)
            users = torch.LongTensor(users)
            mask = data_value.clone()
            mask2 = data_value.clone().cpu() # for only positive samples
            mask[users, negative_samples] = 1
            groundtruth = data_value[mask > 0.]
            pred = prediction [mask > 0.]

            loss, loss_storage = loss_function(pred, groundtruth, drop_rate_schedule(counter))
            user_mapped = torch.tensor([map_dict_old[x.item()] for x in user])
            mask2[mask2 > 0.] = loss_storage.clone().detach().cpu()
            epoch_loss[user_mapped] = mask2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            counter += 1



        print(f"Epoch: {epoch} Train loss: {train_loss}") 
        if epoch%20==0 or epoch >=args.epoch_eval:
            # validation
            best_recall, count = eval(model_test, valid_loader, valid_pos_filtered, train_mat_dense, best_recall, count, epoch_loss, map_dict_old, map_dict_new)

            if count == 10:
                print(f"TEST GROUP {i} DONE!")
                break
print("############################## Training End. ##############################")
