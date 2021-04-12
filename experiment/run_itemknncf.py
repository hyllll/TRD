import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.utils.data as data

import sys


from daisy.model.KNNCFRecommender import ItemKNNCF
from daisy.utils.loader import load_rate, split_test, get_ur
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Item-KNN recommender test')
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
                        default=500, 
                        help='top number of recommend list')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, 
                        help='No. of candidates item for predict')
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
    args = parser.parse_args()


    #temporary used for tuning test result
    train_set1 = pd.read_csv(f'../experiment_data/train1_{args.dataset}_{args.prepro}.dat')
    train_set2 = pd.read_csv(f'../experiment_data/train2_{args.dataset}_{args.prepro}.dat')

    test_set = pd.read_csv(f'../experiment_data/test_{args.dataset}_{args.prepro}.dat')

    train_set1['rating'] = 1.0
    train_set2['rating'] = 1.0
    # validate_set['rating'] = 1.0
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
    
    # get ground truth
    test_ur = get_ur(test_set)
    total_train_ur = get_ur(train_set)

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    print('='*50, '\n')
    # retrain model by the whole train set
    # build recommender model
    model = ItemKNNCF(user_num, item_num, 
                      maxk=args.maxk, 
                      min_k=args.mink, 
                      similarity=args.sim_method)
    model.fit(train_set)
    if not os.path.exists(f'./tmp/{args.dataset}/itemknn/'):
        os.makedirs(f'./tmp/{args.dataset}/itemknn/')
    torch.save(model, f'./tmp/{args.dataset}/itemknn/{args.prepro}_itemknn.pt')

    print('Start Calculating Metrics......')
    # build candidates set
    test_ucands = defaultdict(list)
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        # sub_item_pool = item_pool - v - total_train_ur[k] - validate_ur[k]# remove GT & interacted
        sub_item_pool = item_pool - v - total_train_ur[k]
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
            test_ucands[k] = list(set(samples))
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))

    # get predict result
    preds = {}
    for u in tqdm(test_ucands.keys()):
        pred_rates = [model.predict(u, i) for i in test_ucands[u]]
        rec_idx = np.argsort(pred_rates)[::-1][:args.topk]
        top_n = np.array(test_ucands[u])[rec_idx]
        preds[u] = top_n

    # convert rank list to binary-interaction
    res = preds.copy()
    u_binary = []
    u_result = []
    record = {}
    u_record = {}
    u_test = []
    for u in preds.keys():
        u_record[u] = [u] + res[u].tolist()
        u_result.append(u_record[u])
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
        record[u] = [u] + preds[u]
        u_binary.append(record[u])
        u_test.append(u)

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/itemknn/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # save binary-interaction list to csv file
    pred_csv = pd.DataFrame(data=u_binary)
    pred_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_itemknn.csv', index=False)
    
    res_csv = pd.DataFrame(data=u_result)
    res_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_result_itemknn.csv', index=False)

    res = pd.DataFrame({'metric@K': ['pre', 'rec', 'hr', 'map', 'mrr', 'ndcg']})
    for k in [1, 5, 10, 20, 30, 50]:
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

    res.to_csv(f'{result_save_path}{args.prepro}_itemknn.csv', 
               index=False)
    print('='* 20, ' Done ', '='*20)