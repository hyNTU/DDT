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


from tqdm import tqdm
import toolz
import model
import evaluate
import data_utils
from scipy import sparse

from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, movielens',
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
	default=30,
	help="epoch to start evaluation")
parser.add_argument("--top_k", 
    type = list,
	default= [50, 100],
	help="compute metric @topk")
parser.add_argument("--batch_size", 
    type = int,
	default= 2048,
	help="batch size")
parser.add_argument("--group", 
    type =  int,
	default= 4,
	help="mode of combining embeddings")
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 30000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--model_num', 
	type = int, 
	default = 5,
	help='number of models')
parser.add_argument('--remove', 
	type = float, 
	default = 0.75,
	help='ratio to keep')

args = parser.parse_args()

############################ FUNCTIONS #######################################
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

    return loss_update

def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate


    
########################### Eval #####################################
def eval(model, valid_pos, mat, best_recall, count, model_id):
    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_valid = list(valid_pos.keys())
    for users_valid in toolz.partition_all(15,users_in_valid):
        users_valid = list(users_valid)
        # to find grouping
        group_values = [user_dict[key] for key in users_valid]
        GroundTruth.extend([valid_pos[u] for u in users_valid])
        # repeat each user for n items
        users_valid_torch = torch.tensor(users_valid).repeat_interleave(item_num).cuda()  
        # repeat all items for batch times
        items_full = torch.tensor([i for i in range(item_num)]).repeat(len(users_valid)).cuda()
        group_full = torch.tensor(group_values).repeat_interleave(item_num).cuda() 
        prediction = model(users_valid_torch, items_full, model_id)
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
        torch.save(model.state_dict(), model_path+f'DDT_{args.model}_{args.dataset}.pth')
    else: 
        count += 1
    return best_recall, count

def eval_attn(model, valid_pos, mat, best_recall, count, model_num, attn):
    top_k = args.top_k
    model.eval()
    attn.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_valid = list(valid_pos.keys())
    for users_valid in toolz.partition_all(15,users_in_valid):
        users_valid = list(users_valid)
        # to find grouping
        group_values = [user_dict[key] for key in users_valid]
        GroundTruth.extend([valid_pos[u] for u in users_valid])
        # repeat each user for n items
        users_valid_torch = torch.tensor(users_valid).repeat_interleave(item_num).cuda()  
        # repeat all items for batch times
        items_full = torch.tensor([i for i in range(item_num)]).repeat(len(users_valid)).cuda()
        group_full = torch.tensor(group_values).repeat_interleave(item_num).cuda() 
        predictions = torch.empty((model_num, items_full.shape[0])).cuda()
        for model_id in range(model_num):
            predictions[model_id] = model(users_valid_torch, items_full, model_id)
        prediction = attn(predictions, group_full)
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
        torch.save(attn.state_dict(), model_path+f'DDT_attn_{args.model}_{args.dataset}.pth')
    else: 
        count += 1
    return best_recall, count

########################### Test #####################################
def test(model, test_data_pos, mat, model_num, attn):
    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_test = [key for key in test_data_pos.keys() if key in user_dict]
    for users_test in toolz.partition_all(15,users_in_test):
        users_test = list(users_test)
        group_values = [user_dict[key] for key in users_test]
        GroundTruth.extend([test_data_pos[u] for u in users_test])
        users_test_torch = torch.tensor(users_test).repeat_interleave(item_num).cuda()  
        items_full = torch.tensor([i for i in range(item_num)]).repeat(len(users_test)).cuda()
        group_full = torch.tensor(group_values).repeat_interleave(item_num).cuda()
        predictions = torch.empty((model_num, items_full.shape[0])).cuda()
        for model_id in range(model_num):
            predictions[model_id] = model(users_test_torch, items_full, model_id)
        prediction = attn(predictions, group_full)        
        _, indices = torch.topk(prediction.view(len(users_test),-1)+mat[users_test].cuda()*-9999, max(top_k))
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)


    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


def test_groups(model, test_data_pos, mat, sorted_users, model_num, attn):
    top_k = args.top_k
    model.eval()
    # split_users = torch.tensor_split(sorted_users, 5)
    i = 0
    for group, users in enumerate(sorted_users):
        predictedIndices = []
        GroundTruth = []
        users.cuda()
        new_dict = dict(((key.item(), test_data_pos[key.item()]) for key in users if key.item() in test_data_pos))
        users_in_test = list(new_dict.keys())
        for users_test in toolz.partition_all(15,users_in_test):
            users_test = list(users_test)
            GroundTruth.extend([test_data_pos[u] for u in users_test])
            users_test_torch = torch.tensor(users_test).repeat_interleave(item_num).cuda()   # repeats users_valid for item_num times: 0,0..,1,1..1024,1024..
            items_full = torch.tensor([i for i in range(item_num)]).repeat(len(users_test)).cuda() # this repeat is different from the previous line
            group_full = torch.tensor(group).repeat(len(items_full))
            predictions = torch.empty((model_num, items_full.shape[0])).cuda()
            for model_id in range(model_num):
                predictions[model_id] = model(users_test_torch, items_full, model_id)
            prediction = attn(predictions, group_full)
            _, indices = torch.topk(prediction.view(len(users_test),-1)+mat[users_test].cuda()*-9999, max(top_k))
            indices = indices.cpu().numpy().tolist()
            predictedIndices.extend(indices)
        precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
        
        print(f"################### TEST_group{i} ######################")
        print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
        print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))

        i += 1

################################### END OF FUNCTIONS #####################################

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(args.seed) # cpu
torch.cuda.manual_seed(args.seed) #gpu
np.random.seed(args.seed) #numpy
random.seed(args.seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn


data_path = f'../data/{args.dataset}/'
model_path = f'../models/'

############################## PREPARE DATASET ##########################
######## REWEIGHING FACTOR ########
if args.dataset == 'movielens':
    factor = 1.0
elif args.dataset == 'yelp':
    factor = 0.015
else:
    factor = 0.03

############## USER SORTING #####################
sorted_users = torch.load(f'../data/spec_users_{args.dataset}.pth', weights_only=True)
print("#END OF SORTING")

# to track user in which group
user_dict = {user.item(): index for index, user_list in enumerate(sorted_users) for user in user_list}

model_num = args.model_num

train_data, valid_data, test_data_pos, valid_pos, user_pos, user_num ,item_num, train_mat, train_valid_mat, train_data_noisy = data_utils.load_all(f'{args.dataset}', data_path)

train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, 1, 0, train_data_noisy)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True)

########################### CREATE MODEL #################################
model_test = model.CombinedNetwork(user_num, item_num, 32, 3, 
						0.0, f'{args.model}', model_num)
model_test.cuda()

attention = model.SelfAttention(model_num, 5)
attention.cuda()

optimizer = optim.Adam(model_test.parameters(), lr=0.001)
optimizer_attn = optim.Adam(attention.parameters(), lr=0.0001)



for model_id in range(model_num):      

    ## LOAD HARD & NOISY SAMPLES ##
    # LOAD MAT THAT STORES LOSSES WHEN TRAIN TGT AND ALONE ##
    train_tgt_mat = torch.load(f"../data/overall_loss_{args.model}_{args.dataset}.pth", weights_only=True)
    train_alone_mat = torch.load(f"../data/group_loss_{args.model}_{args.dataset}.pth", weights_only=True)
    train_tgt_mat.requires_grad = False
    train_alone_mat.requires_grad = False

    difference_mat = train_tgt_mat - train_alone_mat


    mask = torch.zeros_like(difference_mat)
    mask[sorted_users[model_id]] = 1.0


    threshold = torch.quantile(difference_mat[sorted_users[model_id]][difference_mat[sorted_users[model_id]]!=0],1.0-args.remove)
    difference_mat *= mask
    difference_mat[difference_mat < threshold] = 0.0
    difference_mat[difference_mat != 0] = 1.0


    ## FILTER THE DATA ##
    train_mat_filtered = train_mat.toarray() - difference_mat.numpy()
    train_data_filtered = []
    for user, data in enumerate(train_mat_filtered):
        for l in np.nonzero(data):
            for item in l:
                train_data_filtered.append([user,item])
    train_mat_filtered = sparse.dok_matrix(train_mat_filtered)
    train_data_noisy_filtered = [1 for _ in range(len(train_mat_filtered))]
    # construct the train and test datasets
    train_loader.dataset.change_data(train_data_filtered, train_mat_filtered, train_data_noisy_filtered)

    print(f"GROUP {model_id} DATA FILTERED")

    train_mat_dense = torch.tensor(train_mat_filtered.toarray()).cpu()


    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    counter = 0
    best_recall = 0.0
    top_k = args.top_k

    for epoch in range(1000):
        model_test.train() # Enable dropout (if have).
        train_loss = 0

        start_time = time.time()
        train_loader.dataset.ng_sample(user_dict) # negative sampling is done here


        for user, item, label, _, group in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()
            group = group.cuda()

            model_test.zero_grad()
            prediction = model_test(user, item, model_id)
            batch_loss = loss_function(prediction, label, drop_rate_schedule(counter))

            if epoch > 2:
                batch_loss *= factor
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss
            counter += 1
        print("epoch: {}, loss:{}".format(epoch,train_loss))
        
        if epoch%20==0 or epoch >= args.epoch_eval:
            best_recall, count = eval(model_test, valid_pos, train_mat_dense, best_recall, count, model_id)
        model_test.train()
        if count == 10:
            print(f"DONE WITH {model_id} model training")
            break
       

print("############################## Training End. ##############################")



print("############################## LOAD MODEL. ##############################")
model_test.load_state_dict(torch.load(model_path+f'DDT_{args.model}_{args.dataset}.pth'))
model_test.cuda()

for param in model_test.parameters():
    param.requires_grad = False

train_mat_filtered = train_mat.toarray()
train_data_filtered = []
for user, data in enumerate(train_mat_filtered):
    for l in np.nonzero(data):
        for item in l:
            train_data_filtered.append([user,item])
train_mat_filtered = sparse.dok_matrix(train_mat_filtered)
train_data_noisy_filtered = [1 for _ in range(len(train_mat_filtered))]
train_loader.dataset.change_data(train_data_filtered, train_mat_filtered, train_data_noisy_filtered)

train_mat_dense = torch.tensor(train_mat_filtered.toarray()).cpu()

print("TRAINING ATTENTION")

count, best_hr = 0, 0
counter = 0
best_recall = 0.0
top_k = args.top_k

for epoch in range(100):
    train_loss = 0

    start_time = time.time()
    train_loader.dataset.ng_sample(user_dict) # negative sampling is done here


    for user, item, label, _, group in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()
        group = group.cuda()
        attention.zero_grad()
        predictions = torch.empty((model_num, label.shape[0])).cuda()
        for i in range(model_num):
            predictions[i] = model_test(user.cuda(),item.cuda(), i)
        prediction = attention(predictions, group)
        batch_loss = loss_function(prediction, label, drop_rate_schedule(counter))


        batch_loss.backward()
        optimizer_attn.step()
        train_loss += batch_loss
        counter += 1
    print("epoch: {}, loss:{}".format(epoch,train_loss))

    best_recall, count = eval_attn(model_test, valid_pos, train_mat_dense, best_recall, count, model_num, attention)
    # model_test.train()
    attention.train()
    if count == 5:
        break


print("############################## Training End. ##############################")
attention.load_state_dict(torch.load(model_path+f'DDT_attn_{args.model}_{args.dataset}.pth'))
attention.cuda()

    ########################### TEST #####################################

train_valid_mat_dense = torch.tensor(train_valid_mat.toarray()).cpu() 
test(model_test, test_data_pos, train_valid_mat_dense, model_num, attention)
test_groups(model_test, test_data_pos, train_valid_mat_dense, sorted_users, model_num, attention)

