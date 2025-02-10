import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy import sparse

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
parser.add_argument('--model', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--seed', 
	type = int,
	help = 'seed for reproducibility',
	default = 2019)
parser.add_argument("--gpu", 
	type=str,
	default="0",
	help="gpu card ID")
parser.add_argument("--epoch_eval", 
    type = int,
	default=1,
	help="epoch to start evaluation")
parser.add_argument("--top_k", 
    type = list,
	default= [50, 100],
	help="compute metric @topk")
parser.add_argument("--batch_size", 
    type = int,
	default= 2048,
	help="batch size")
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 6000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--exponent', 
	type = float, 
	default = 1, 
	help='exponent of the drop rate {0.5, 1, 2}')
parser.add_argument('--model_num', 
	type = int, 
	default = 5, 
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

train_data, valid_data, test_data_pos, valid_pos, user_pos, user_num ,item_num, train_mat, train_valid_mat, train_data_noisy = data_utils.load_all(f'{args.dataset}', data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, 1, 0, train_data_noisy)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data_pos)))



########################### Temp Softmax #####################################
def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update, loss_mul

def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate
     
########################### Eval #####################################
def eval(model, valid_pos, mat, best_recall, count, epoch_loss, map_dict):
    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_valid = list(valid_pos.keys())
    for users_valid in toolz.partition_all(15,users_in_valid):
        users_valid_sub = [map_dict[x] for x in users_valid]
        users_valid = list(users_valid)
        GroundTruth.extend([valid_pos[u] for u in users_valid])
        users_valid_torch = torch.tensor(users_valid_sub).repeat_interleave(item_num).cuda()   # repeats users_valid for item_num times: 0,0..,1,1..1024,1024..
        items_full = torch.tensor([i for i in range(item_num)]).repeat(len(users_valid)).cuda() # this repeat is different from the previous line
        prediction = model(users_valid_torch, items_full)
        _, indices = torch.topk(prediction.view(len(users_valid),-1)+mat[users_valid].cuda()*-9999, max(top_k))
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    epoch_recall = recall[0]

    print("################### EVAL ######################")
    print(f"Recall:{recall} NDCG: {NDCG}")
    if epoch_recall > best_recall:
        best_recall = epoch_recall
        count = 0
        torch.save(epoch_loss, f'../data/group_loss_{args.model}_{args.dataset}.pth')
    else: 
        count += 1
    return best_recall, count



########################### BEGIN USER LOOP ###########################
list_of_users = torch.load(f'../data/spec_users_{args.dataset}.pth', weights_only=True)
print("LOADED USERS")
train_mat_dense = torch.tensor(train_mat.toarray()).cpu()
train_mat_sum = train_mat_dense.sum(1)

epoch_loss = torch.zeros_like(train_mat_dense)


# need to change valid_pos

for i, group in enumerate(list_of_users):

    # filtering data
    train_mat_filtered = train_mat_dense[group].numpy()
    
    train_data_filtered = []
    for user, data in enumerate(train_mat_filtered):
        for l in np.nonzero(data):
            for item in l:
                train_data_filtered.append([user,item])
    train_mat_filtered = sparse.dok_matrix(train_mat_filtered)
    train_data_noisy_filtered = [1 for _ in range(len(train_mat_filtered))]


    user_num = group.shape[0]
    train_loader.dataset.change_data(train_data_filtered, train_mat_filtered, train_data_noisy_filtered)
    valid_pos_filtered = {key.item(): valid_pos[key.item()] for key in group if key.item() in valid_pos}
    test_data_pos_filtered = {key.item(): test_data_pos[key.item()] for key in group if key.item() in test_data_pos}
    print(f"DATA FILTERED FOR GROUP {i}")
    # to map old user values to new user values
    map_dict_old = {i : key.item() for i,key in enumerate(group)}
    map_dict_new = {key.item(): i for i,key in enumerate(group)}
########################### CREATE MODEL #################################

    model_test = model.NCF(user_num, item_num, 32, 3, 
                            0.0, f'{args.model}')

    model_test.cuda()
    BCE_loss = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model_test.parameters(), lr=0.001)

########################### TRAINING #####################################
    count, best_hr = 0, 0
    best_recall = 0.0
    top_k = args.top_k
    counter = 0

    for epoch in range(1000):
        model_test.train() # Enable dropout (if have).
        train_loss = 0

        start_time = time.time()
        train_loader.dataset.ng_sample() # negative sampling is done here     

        for user, item, label, _, _ in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            user_mapped = torch.tensor([map_dict_old[x.item()] for x in user]).cuda()
            batch_pos_user = user_mapped[label > 0.]
            batch_pos_item = item[label > 0.]

            model_test.zero_grad()
            prediction = model_test(user, item) # size 1024
            loss, loss_storage = loss_function(prediction, label, drop_rate_schedule(counter))

            epoch_loss[batch_pos_user.cpu(), batch_pos_item.cpu()] = loss_storage[label > 0.].cpu()
            
            loss.backward()
            optimizer.step()
            train_loss += loss
            counter += 1

        print("epoch: {}, loss:{}".format(epoch,train_loss))

        if epoch >= args.epoch_eval:
            best_recall, count = eval(model_test, valid_pos_filtered, train_mat_dense, best_recall, count, epoch_loss, map_dict_new)
        model_test.train()
        if count == 10:

            break


print("############################## Training End. ##############################")

