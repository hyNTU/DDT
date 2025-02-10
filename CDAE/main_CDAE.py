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
import CDAE_utils
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
parser.add_argument("--model_num", 
    type = int,
	default= 5,
	help="number of models")
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
parser.add_argument('--remove', 
	type = float, 
	default = 0.75,
	help='how many epochs to linearly increase drop_rate')


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
def eval(model, valid_loader, valid_data_pos, train_mat, best_recall, count, model_id):
    top_k = args.top_k
    model.eval()
    # model prediction can be more efficient instead of looping through each user, do it by batch
    predictedIndices_all = torch.empty(user_num, top_k[-1], dtype=torch.long) # predictions
    GroundTruth = list(valid_data_pos.values()) # ground truth is exact item indices
    for user_valid, data_value_valid in valid_loader:
        with torch.no_grad():
            user_valid = user_valid.cuda()
            prediction_input_from_train = torch.tensor(train_mat[user_valid.cpu()]).cuda()
            prediction = model(user_valid, prediction_input_from_train, model_id) # prediction of the batch from train matrix
            valid_data_mask = train_mat[user_valid.cpu()] * -9999# depends on the size of data

            prediction = prediction + torch.tensor(valid_data_mask).float().cuda()
            _, indices = torch.topk(prediction, top_k[-1])
            predictedIndices_all[user_valid.cpu()] = indices.cpu()

    predictedIndices = predictedIndices_all[list(valid_data_pos.keys())]
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    print(f"Recall:{recall} NDCG: {NDCG}")
    if recall[0] > best_recall:
        best_recall = recall[0]
        count = 0
        torch.save(model.state_dict(), model_path+f'DDT_CDAE_{args.dataset}.pth')
    else: 
        count += 1
    return best_recall, count


def eval_attn(model, valid_loader, valid_data_pos, train_mat, best_recall, count, model_num, attn):
    top_k = args.top_k
    model.eval()
    attn.eval()
    # model prediction can be more efficient instead of looping through each user, do it by batch
    predictedIndices_all = torch.empty(user_num, top_k[-1], dtype=torch.long) # predictions
    GroundTruth = list(valid_data_pos.values()) # ground truth is exact item indices
    for user_valid, data_value_valid in valid_loader:
        with torch.no_grad():
            user_valid = user_valid.cuda()
            prediction_input_from_train = torch.tensor(train_mat[user_valid.cpu()]).cuda() # 4096, 57396
            predictions = torch.empty((model_num, user_valid.shape[0], item_num)).cuda()
            group = torch.tensor([user_dict[x.item()] for x in user_valid]).cuda() # 4096
            for model_id in range(model_num):
                predictions[model_id] = model(user_valid, prediction_input_from_train, model_id) # prediction of the batch from train matrix
            # print(predictions.shape, group.shape)
            prediction = attn.massforward(predictions, group)
            valid_data_mask = train_mat[user_valid.cpu()] * -9999# depends on the size of data

            prediction = prediction + torch.tensor(valid_data_mask).float().cuda()
            _, indices = torch.topk(prediction, top_k[-1])
            predictedIndices_all[user_valid.cpu()] = indices.cpu()

    predictedIndices = predictedIndices_all[list(valid_data_pos.keys())]
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    print(f"Recall:{recall} NDCG: {NDCG}")
    if recall[0] > best_recall:
        best_recall = recall[0]
        count = 0
        torch.save(attn.state_dict(), model_path+f'DDT_attn_CDAE_{args.dataset}.pth')
    else: 
        count += 1
    return best_recall, count



########################### Test #####################################
def test(model, test_data_pos, train_mat, valid_mat, model_num, attn):
    top_k = args.top_k
    model.eval()
    attn.eval()
    predictedIndices = [] # predictions
    GroundTruth = list(test_data_pos.values())

    for users in toolz.partition_all(1000, list(test_data_pos.keys())): # looping through users in test set
        user_id = torch.tensor(list(users)).cuda()
        data_value_test = torch.tensor(train_mat[list(users)]).cuda()
        predictions = torch.empty((model_num, user_id.shape[0], item_num)).cuda()
        group = torch.tensor([user_dict[x.item()] for x in user_id]).cuda()
        for model_id in range(model_num):
            predictions[model_id] = model(user_id, data_value_test, model_id)
        prediction = attn.massforward(predictions, group)
        test_data_mask = (train_mat[list(users)] + valid_mat[list(users)]) * -9999

        prediction = prediction + torch.tensor(test_data_mask).float().cuda()
        _, indices = torch.topk(prediction, top_k[-1]) # returns sorted index based on highest probability
        indices = indices.cpu().numpy().tolist()
        predictedIndices += indices # a list of top 100 predicted indices

    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


def test_groups(model, test_data_pos, train_mat, valid_mat, sorted_users, model_num, attn):
    top_k = args.top_k
    model.eval()
    i = 0
    for users_1 in sorted_users:
        users_1.cuda()
        new_dict = dict( ((key.item(), test_data_pos[key.item()]) for key in users_1 if key.item() in test_data_pos))
        predictedIndices = [] # predictions
        GroundTruth = list(new_dict.values())

        for users in toolz.partition_all(1000, list(new_dict.keys())): # looping through users in test set
            user_id = torch.tensor(list(users)).cuda()
            data_value_test = torch.tensor(train_mat[list(users)]).cuda()
            predictions = torch.empty((model_num, user_id.shape[0], item_num)).cuda()
            group = torch.tensor([user_dict[x.item()] for x in user_id]).cuda()
            for model_id in range(model_num):
                predictions[model_id] = model(user_id, data_value_test, model_id)
            prediction = attn.massforward(predictions, group)
            test_data_mask = (train_mat[list(users)] + valid_mat[list(users)]) * -9999

            prediction = prediction + torch.tensor(test_data_mask).float().cuda()
            _, indices = torch.topk(prediction, top_k[-1]) # returns sorted index based on highest probability
            indices = indices.cpu().numpy().tolist()
            predictedIndices += indices # a list of top 100 predicted indices

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

train_data, valid_data, test_data_pos, valid_pos, user_pos, user_num ,item_num, train_mat, train_valid_mat, train_data_noisy = CDAE_utils.load_all(f'{args.dataset}', data_path)

_, _, train_data_pos, valid_data_pos, test_data_pos, _, _ , _, _, valid_mat, _ = data_utils.load_all(f'{args.dataset}', data_path)


train_dataset = CDAE_utils.NCFData(
		train_data, item_num, train_mat, 1, 0, train_data_noisy)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True)
        
users_list = np.array([i for i in range(user_num)])

valid_mat_dense = valid_mat.toarray()

valid_dataset = data_utils.DenseMatrixUsers(users_list, valid_mat_dense)
valid_loader = data.DataLoader(valid_dataset, batch_size=4096, shuffle=True)

########################### CREATE MODEL #################################
model_test = model.CombinedNetwork(user_num, item_num, 32, 0.2, model_num)

model_test.cuda()

attention = model.SelfAttention(model_num, 5)
attention.cuda()

optimizer = optim.Adam(model_test.parameters(), lr=0.001)
optimizer_attn = optim.Adam(attention.parameters(), lr=0.0001)

for model_id in range(model_num):      

    difference_mat = torch.load(f'../data/overall_loss_CDAE_{args.dataset}.pth', weights_only=True) - torch.load(f"../data/group_loss_CDAE_{args.dataset}.pth", weights_only=True)
    difference_mat.requires_grad =False

    mask = torch.zeros_like(difference_mat)
    mask[sorted_users[model_id]] = 1.0


    threshold = torch.quantile(difference_mat[sorted_users[model_id]][difference_mat[sorted_users[model_id]]!=0], 1.0-args.remove)

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
            prediction = model_test(user, train_mat_dense[user.cpu()].cuda(), model_id)
            prediction = prediction[torch.arange(item.shape[0]), item]
            batch_loss = loss_function(prediction, label, drop_rate_schedule(counter))

            if epoch > 2:
                batch_loss *= factor
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss
            counter += 1
        print("epoch: {}, loss:{}".format(epoch,train_loss))
        
        if epoch%20==0 or epoch >= args.epoch_eval:
            best_recall, count = eval(model_test, valid_loader, valid_data_pos, train_mat_filtered.toarray(), best_recall, count, model_id)
        model_test.train()
        if count == 10:
            print(f"DONE WITH {model_id} model training")
            break
       

print("############################## Training End. ##############################")



print("############################## LOAD MODEL. ##############################")
model_test.load_state_dict(torch.load(model_path+f'DDT_CDAE_{args.dataset}.pth'))
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

for epoch in range(1000):
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
            placeholder = model_test(user,train_mat_dense[user.cpu()].cuda(), i)
            predictions[i] = placeholder[torch.arange(item.shape[0]), item]

        prediction = attention(predictions, group)
        batch_loss = loss_function(prediction, label, drop_rate_schedule(counter))


        batch_loss.backward()
        optimizer_attn.step()
        train_loss += batch_loss
        counter += 1
    print("epoch: {}, loss:{}".format(epoch,train_loss))
    best_recall, count = eval_attn(model_test, valid_loader, valid_data_pos, train_mat_dense, best_recall, count, model_num, attention)

    attention.train()
    if count == 5:
        break


print("############################## Training End. ##############################")
attention.load_state_dict(torch.load(model_path+f'DDT_attn_CDAE_{args.dataset}.pth'))
attention.cuda()

    ########################### TEST #####################################

test(model_test, test_data_pos, train_mat_dense, valid_mat_dense, model_num, attention)
test_groups(model_test, test_data_pos, train_mat_dense, valid_mat_dense, sorted_users, model_num, attention)

