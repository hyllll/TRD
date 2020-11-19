import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import math
import gc
import os
import argparse
import torch.backends.cudnn as cudnn

import sys
# sys.path.append('/home/xinghua/Hongyang/Code-submit')
sys.path.append('/home/workshop/lhy/code-submit')

from collections import defaultdict
# from memory_profiler import profile
from tqdm import tqdm
from daisy.model.KNNCFRecommender import ItemKNNCF
# from daisy.model.TRDRecommender import TRD, get_next_state, get_reward, create_initial_state
from daisy.utils.loader import load_rate, split_test, get_ur, PairMFData, get_ur_l
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

class Embed(nn.Module):
    def __init__(self, user_num, item_num, factor_num, pretrain_model, pretrain):
        super(Embed, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num

        self.embed_user = nn.Embedding(self.user_num, self.factor_num)
        self.embed_item = nn.Embedding(self.item_num + 1, self.factor_num, padding_idx=self.item_num)

        if pretrain:
            if method == 'Item2Vec':
                item_weights = np.random.normal(size=(item_num + 1, factor_num), scale=0.01)
                for k, v in model.item2vec.items():
                    if isinstance(k, int):
                        item_weights[k] = v
                self.embed_item.weight.data.copy_(torch.from_numpy(item_weights))

                user_weights = np.random.normal(size=(user_num, factor_num), scale=0.01)
                for k, v in model.user_vec_dict.items():
                    if isinstance(k, int):
                        user_weights[k] = v
                self.embed_user.weight.data.copy_(torch.from_numpy(user_weights))
                print("embedding load completed")
                del item_weights, user_weights

            if method == 'bprmf':
                weight = model.embed_item.weight.cpu().detach()
                pad = np.random.normal(size=(1, factor_num), scale=0.01)
                pad = torch.FloatTensor(pad)
                weight = torch.cat([weight, pad])
                self.embed_item.weight.data.copy_(weight) 

                weight = model.embed_user.weight.cpu().detach()
                self.embed_user.weight.data.copy_(weight)
                print("embedding load completed")
        else:
            print("not load item2vec")
            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.normal_(self.embed_item.weight, std=0.01)              

class Actor(nn.Module):
    def __init__(self, embedding, factor_num=32, gpuid='0'):
        super(Actor, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.factor_num = factor_num
        self.embed = embedding

        self.fc1 = nn.Linear(self.factor_num * 6, 256)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128,self.factor_num)
        self.out.weight.data.normal_(0,0.1)
    
    def forward(self, x0, x1):
        x0 = self.embed.embed_user(torch.tensor(x0).cuda().long()).view(-1, self.factor_num)
        x1 = self.embed.embed_item(torch.tensor(x1).cuda().long()).view(-1, self.factor_num * 5)
        x = torch.cat([x0,x1], dim=1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(x)
        action = self.out(x)
        return action

class Critic(nn.Module):
    def __init__(self, embedding, factor_num=32, gpuid='0'):
        super(Critic, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.factor_num = factor_num
        self.embed = embedding

        self.fc1 = nn.Linear(self.factor_num * 7, 256)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(256,128)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128,1)
        self.out.weight.data.normal_(0,0.1)
    
    def forward(self, x0, x1, x2):
        x0 = self.embed.embed_user(torch.tensor(x0).cuda().long()).view(-1, self.factor_num)
        x1 = self.embed.embed_item(torch.tensor(x1).cuda().long()).view(-1, self.factor_num * 5)
        # x2 = self.embed_item(torch.tensor(x2).cuda().long()).view(-1, self.factor_num)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(x)
        value = self.out(x)
        return value


class DDPG():
    def __init__(self, user_num, item_num, n_actions, pretrain_model, method, lr=0.01, epsilon=0.9, gamma=0.9, memory_capacity=20, iteration=5, batch_size=8, factor_num=32, gpuid='0', pretrain=1, use_cuda=True):
        super(DDPG, self).__init__()

        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.memory_capacity = memory_capacity
        self.iteration = iteration
        self.batch_size = batch_size
        self.factor_num = factor_num
        self.n_states = 5
        self.tau = 0.001

        self.embed = Embed(user_num, item_num, factor_num, pretrain_model, pretrain, method)
        self.actor_eval = Actor(self.embed, factor_num)
        self.actor_target = Actor(self.embed, factor_num)
        self.critic_eval = Critic(self.embed, factor_num)
        self.critic_target = Critic(self.embed, factor_num)


        self.use_cuda = use_cuda
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0      # for storing memory

        self.memory = np.zeros((self.memory_capacity, self.n_states * 2 + 2 + self.factor_num))
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.action_space = []
        self.candidate_actions = []

        self.loss_func = nn.MSELoss()

        if torch.cuda.is_available():
            self.actor_eval.cuda()
            self.actor_target.cuda()
            self.critic_eval.cuda()
            self.critic_target.cuda()
            self.embed.cuda()
            self.loss_func.cuda()
    
    def create_action_space(self, u_action_space):
        self.action_space = u_action_space[:self.n_actions]
        self.candidate_actions = u_action_space[self.n_actions:]
    
    def update_action_space(self, action):
        self.action_space.remove(action)
        add = self.candidate_actions.pop(0)
        self.action_space.append(add)


    def choose_action(self, user, state, model):
        # user = self.embed.embed_user(torch.tensor(user).cuda().long()).view(-1, self.factor_num)
        # state = self.embed.embed_item(torch.tensor(state).cuda().long()).view(-1, self.factor_num * 5)
        action = self.actor_eval(user, state)
        item_embedding = []
        for i in self.action_space:
            item_embedding.append(np.array(model.item2vec[i]))
        item_embedding =  torch.Tensor(item_embedding, )
        items = item_embedding.t()
        items = items.unsqueeze(0)
        action_embed = torch.reshape(action, [1,32]).unsqueeze(0)
        items = items.cuda()
        m = torch.bmm(action_embed,items).squeeze(0)
        sorted_m,indices = torch.sort(m,descending=True)
        index_list = list(indices[0])
        item = self.action_space[index_list[0].cpu().numpy()]
        return item, action
    
    """a转换为array"""
    def store_transition(self, u, s, a, r, s_):
        u = np.array(u)
        s = np.array(s)
        s_ = np.array(s_)

        a = torch.squeeze(a)
        a = a.cpu()
        a = a.detach().numpy()

        transition = np.hstack((u, s, a, r, s_))   # horizon add
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_u = b_memory[:, :1].astype(int)
        b_s = torch.FloatTensor(b_memory[:, 1:self.n_states+1])
        b_a = torch.FloatTensor(b_memory[:, self.n_states+1: self.n_states + 33])
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 33: self.n_states + 34])
        b_s_ = torch.FloatTensor(b_memory[:, self.n_states + 34:self.n_states + 35 +self.n_states])
        
        # user = self.embed.embed_user(torch.tensor(b_u).cuda().long()).view(-1, self.factor_num)
        # state = self.embed.embed_item(torch.tensor(b_s).cuda().long()).view(-1, self.factor_num * 5)
        # next_state = self.embed.embed_item(torch.tensor(b_s_).cuda().long()).view(-1, self.factor_num * 5)
        if torch.cuda.is_available():
            b_r = b_r.cuda()
            b_a = b_a.cuda()

        actor_loss = -self.critic_eval(b_u, b_s, self.actor_eval(b_u, b_s)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        target_Q = self.critic_target(b_u,  b_s_, self.actor_target(b_u, b_s_))
        target_Q = b_r + (self.gamma * target_Q).detach()
        current_Q = self.critic_eval(b_u, b_s, b_a)
        
        critic_loss = self.loss_func(target_Q, current_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()



        for param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.learn_step_counter += 1

def get_next_state(state, action, t_ur):
    if action in t_ur:
        state.pop(0)
        state.append(action)
        s_next = state
    else:
        s_next = state
    return s_next 

def get_reward(rec_list, action, t_ur):
    if action in t_ur:
        rel = 1
        r = np.subtract(np.power(2, rel), 1) / np.log2(len(rec_list) + 1)
    else:
        rel = 0
        r = 0
    return r

def pad_ur(ur, item_num):
    user_record = ur
    for _ in range(5 - len(ur)):
        user_record.insert(0, item_num)
    return user_record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG recommender test')
    # common settings
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset preprocess op.: origin/5core/10core')
    parser.add_argument('--topk', 
                        type=int, 
                        default=50, 
                        help='top number of recommend list')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, 
                        help='No. of candidates item for predict')
    parser.add_argument('--method', 
                        type=str, 
                        default='bprmf', 
                        help='bprmf, Item2Vec')
    # algo settings
    parser.add_argument('--sim_method', 
                        type=str, 
                        default='cosine', 
                        help='method to calculate similarity, options: cosine/jaccard/pearson')
    parser.add_argument('--maxk', 
                        type=int, 
                        default=40, 
                        help='The (max) number of neighbors to take into account')
    parser.add_argument('--mink', 
                        type=int, 
                        default=1, 
                        help='The (min) number of neighbors to take into account')
    parser.add_argument('--n_actions', 
                        type=int, 
                        default=20,
                        help='action space')
    parser.add_argument('--memory_capacity', 
                        type=int, 
                        default=20,
                        help='memory_capacity')
    parser.add_argument('--pretrain',
                        type=int,
                        default=0, 
                        help='whether use pretrain vector')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    args = parser.parse_args()

    '''Test Process for Metrics Exporting'''
    train_set1 = pd.read_csv(f'../experiment_data/train1_{args.dataset}_{args.prepro}.dat')
    train_set2 = pd.read_csv(f'../experiment_data/train2_{args.dataset}_{args.prepro}.dat')

    test_set = pd.read_csv(f'../experiment_data/test_{args.dataset}_{args.prepro}.dat')

    train_set1['rating'] = 1.0
    train_set2['rating'] = 1.0
    test_set['rating'] = 1.0

    train_set = pd.concat([train_set1, train_set2], ignore_index=True)

    split_idx = len(train_set)
    df = pd.concat([train_set, test_set], ignore_index=True)
    df['user'] = pd.Categorical(df['user']).codes
    df['item'] = pd.Categorical(df['item']).codes

    train_set = df.iloc[ : split_idx, :].copy()
    test_set = df.iloc[split_idx: , :].copy()

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()
    print(user_num, item_num)

    test_ur= get_ur(test_set)
    total_train_ur = get_ur(train_set)

    test_ur_l = get_ur_l(test_set)
    total_train_ur_l = get_ur_l(train_set)


    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    print('='*50, '\n')
    model = ItemKNNCF(user_num, item_num, 
                      maxk=args.maxk, 
                      min_k=args.mink, 
                      similarity=args.sim_method)
    model.fit(train_set)

    similarity_matrix = model.W_sparse.toarray()
    
    #construct action space
    u_action_space_train = {}
    u_action_space_test = {}
    u_init_train = {}
    u_init_test = {}
    user_set = set()
    for k, v in total_train_ur_l.items():
        user_set.add(k)
        u_init_train[k] = total_train_ur_l[k][0:5]
        u_init_test[k] = total_train_ur_l[k][-5:]
        index_train_list = []
        index_test_list = []
        for i in u_init_train[k]:
            index = similarity_matrix[i].argsort()[-150:][::-1].tolist()
            index_train_list.append(index)
        for i in u_init_test[k]:
            index = similarity_matrix[i].argsort()[-150:][::-1].tolist()
            index_test_list.append(index)
        if len(index_train_list) == 5:
            action = list(zip(index_train_list[4], index_train_list[3], index_train_list[2], index_train_list[1], index_train_list[0]))
        elif len(index_train_list) == 4:
            action = list(zip(index_train_list[3], index_train_list[2], index_train_list[1], index_train_list[0]))
        elif len(index_train_list) == 3:
            action = list(zip(index_train_list[2], index_train_list[1], index_train_list[0]))
        elif len(index_train_list) == 2:
            action = list(zip(index_train_list[1], index_train_list[0]))
        elif len(index_train_list) == 1:
            action = index_train_list[0]
        space = []
        if len(index_train_list) != 1:
            for t in action:
                space += t
            u_action_space_train[k] = list(set(space))
            u_action_space_train[k].sort(key=space.index)
        else:
            u_action_space_train[k] = action

        if len(index_test_list) == 5:
            action = list(zip(index_test_list[4], index_test_list[3], index_test_list[2], index_test_list[1], index_test_list[0]))
        elif len(index_test_list) == 4:
            action = list(zip(index_test_list[3], index_test_list[2], index_test_list[1], index_test_list[0]))
        elif len(index_test_list) == 3:
            action = list(zip(index_test_list[2], index_test_list[1], index_test_list[0]))
        elif len(index_test_list) == 2:
            action = list(zip(index_test_list[1], index_test_list[0]))
        elif len(index_test_list) == 1:
            action = index_test_list[0]
        space = []
        if len(index_test_list) != 1:
            for t in action:
                space += t
            u_action_space_test[k] = list(set(space))
            u_action_space_test[k].sort(key=space.index)
        else:
            u_action_space_test[k] = action

    del u_init_train, u_init_test, space
    gc.collect()
    print("action space complete")

    test_ucands = defaultdict(list)
    user_test_set = set()
    for k, v in test_ur.items():
        user_test_set.add(k)
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        sub_item_pool = item_pool - v - total_train_ur[k] # remove GT & interacted
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))
    
    if args.method == 'Item2Vec':
        model = torch.load(f'./tmp/{args.dataset}/Item2Vec/{args.prepro}_Item2Vec.pt')
    else:
        model = torch.load(f'./tmp/{args.dataset}/bprmf/{args.prepro}_bprmf.pt')
    ddpg = DDPG(user_num,  item_num, args.n_actions, model, args.method, gpuid=args.gpu, pretrain=args.pretrain)
    print("=======model initial completed========")
    preds = {}
    epoch = 0
    total_ep_r = 0
    print("======training=======")
    for user in tqdm(user_set):
        ep_r = 0
        ur = total_train_ur_l[user][0:5]
        s = pad_ur(ur, item_num)
        recommend_item = []
        ddpg.action_space = []
        ddpg.candidate_actions = []
        ddpg.create_action_space(u_action_space_train[user])
        ddpg.memory_counter = 0
        for t in range(50):
            a, action_emb = ddpg.choose_action(user, s, model)
            recommend_item.append(a)
            ddpg.update_action_space(a)
            s_ = get_next_state(s, a, total_train_ur[user])
            r = get_reward(recommend_item, a, total_train_ur[user])
            ddpg.store_transition(user, s, action_emb, r, s_)
            ep_r += r
            if ddpg.memory_counter > args.memory_capacity:
                ddpg.learn()
            s = s_
        preds[user] = recommend_item
        gc.collect()
        epoch += 1
        total_ep_r += ep_r
        if epoch % 50 == 0:
            # print(total_ep_r / 50)
            total_ep_r = 0
    del preds            

    print("=====testing=====")
    preds = {}
    user_test = set()
    for user in tqdm(user_test_set):
        ur = total_train_ur_l[user][-5:]
        s = pad_ur(ur, item_num)
        recommend_item = []
        if not user in u_action_space_test.keys():
            continue
        else:
            user_test.add(user)
        ddpg.create_action_space(u_action_space_test[user])
        for t in range(20):
            a, _ = ddpg.choose_action(user, s, model)
            recommend_item.append(a)
            ddpg.update_action_space(a)
            s_ = get_next_state(s, a, test_ur[user])
            s = s_
        preds[user] = recommend_item

    u_binary = []
    u_result = []
    res = preds.copy()
    record = {}
    u_record = {}
    u_test = []
    for u in user_test:
        u_test.append(u)
        u_record[u] = [u] + res[u]
        u_result.append(u_record[u])
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
        record[u] = [u] + preds[u]
        u_binary.append(record[u])
    
    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/ddpg/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    
    # save binary-interaction list to csv file
    pred_csv = pd.DataFrame(data=u_binary)
    pred_csv.to_csv(f'{result_save_path}{args.dataset}_rl_{args.prepro}_{args.pretrain}_{args.method}.csv', index=False)

    test_user = pd.DataFrame(data=u_result)
    test_user.to_csv(f'{result_save_path}{args.dataset}_rl_{args.prepro}_{args.pretrain}_{args.method}_testuser.csv', index=False)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})

    for k in [1, 5, 10]:
        if k > args.topk:
            continue
        tmp_preds = preds.copy()        
        tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

        pre_k = np.mean([precision_at_k(r, k) for r in tmp_preds.values()])
        rec_k = recall_at_k(tmp_preds, test_ur, u_test, k)
        hr_k = hr_at_k(tmp_preds, test_ur)
        map_k = map_at_k(tmp_preds.values())
        mrr_k = mrr_at_k(tmp_preds, k)
        ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

        if k == 10:
            print(f'Precision@{k}: {pre_k:.4f}')
            print(f'Recall@{k}: {rec_k:.4f}')
            print(f'HR@{k}: {hr_k:.4f}')
            print(f'MAP@{k}: {map_k:.4f}')
            print(f'MRR@{k}: {mrr_k:.4f}')
            print(f'NDCG@{k}: {ndcg_k:.4f}')

        res[k] = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])

    res.to_csv(f'{result_save_path}{args.dataset}_rl_{args.n_actions}_{args.prepro}_{args.pretrain}_{args.method}.csv', 
               index=False)
    print('='* 20, ' Done ', '='*20)


    
            






