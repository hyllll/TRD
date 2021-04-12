import os
import logging
import logging.config
import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import sys


from daisy.model.MostPopRecommender import MostPop
from daisy.utils.loader import load_rate, split_test, get_ur
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Most-Popular recommender test')
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
                        default=200, 
                        help='top number of recommend list')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, help='No. of candidates item for predict')
    # algo settings
    parser.add_argument('--pop_n', 
                        type=int, 
                        default=1000, 
                        help='Initial selected number of Most-popular')
    args = parser.parse_args()


    train_set1 = pd.read_csv(f'../experiment_data/train1_{args.dataset}_{args.prepro}.dat')
    train_set2 = pd.read_csv(f'../experiment_data/train2_{args.dataset}_{args.prepro}.dat')

    test_set = pd.read_csv(f'../experiment_data/test_{args.dataset}_{args.prepro}.dat')

    train_set1['rating'] = 1.0
    train_set2['rating'] = 1.0
    # validate_set['rating'] = 1.0
    test_set['rating'] = 1.0
    train_set = pd.concat([train_set1, train_set2], ignore_index=True)
    # df = pd.concat([train_set, validate_set, test_set], ignore_index=True)
    
    split_idx_1 = len(train_set1)
    split_idx_2 = len(train_set2) + split_idx_1

    df = pd.concat([train_set,  test_set], ignore_index=True)
    df['user'] = pd.Categorical(df['user']).codes
    df['item'] = pd.Categorical(df['item']).codes

    train_set1, train_set2, test_set = df.iloc[:split_idx_1, :].copy(), df.iloc[split_idx_1:split_idx_2, :].copy(), df.iloc[split_idx_2:, :].copy()
    train_set = pd.concat([train_set1,  train_set2], ignore_index=True)
    
    user_num = df['user'].nunique()
    item_num = df['item'].nunique()   
    print(user_num, item_num)

    test_ur = get_ur(test_set)
    train1_ur = get_ur(train_set1)
    train2_ur = get_ur(train_set2)

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    model = MostPop(args.pop_n)
    model.fit(train_set1)
    preds = model.predict(train2_ur, train1_ur, args.topk)

    if not os.path.exists(f'./tmp/{args.dataset}/mostpop/'):
        os.makedirs(f'./tmp/{args.dataset}/mostpop/')
    
    # convert rank list to binary-interaction
    res = preds.copy()
    u_binary = []
    u_result = []
    record = {}
    u_record = {}
    u_test = []
    for u in preds.keys():
        u_record[u] = [u] + res[u]
        u_result.append(u_record[u])
        preds[u] = [1 if i in train2_ur[u] else 0 for i in preds[u]]
        record[u] = [u] + preds[u]
        u_binary.append(record[u])
        u_test.append(u)
    
    mostpop_train = model.get_pop_value(train_set1, res)
    model.save(mostpop_train, f'./tmp/{args.dataset}/mostpop/mostpop_train')

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/mostpop/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    
    # save binary-interaction list to csv file
    pred_csv = pd.DataFrame(data=u_binary)
    pred_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_mostpop_train.csv', index=False)
    
    res_csv = pd.DataFrame(data=u_result)
    res_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_result_mostpop_train.csv', index=False)

    """get test result"""
    preds = model.predict(test_ur, train1_ur, args.topk)
    res = preds.copy()
    u_binary = []
    u_result = []
    record = {}
    u_record = {}
    for u in preds.keys():
        u_record[u] = [u] + res[u]
        u_result.append(u_record[u])
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
        record[u] = [u] + preds[u]
        u_binary.append(record[u])

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/mostpop/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    
    # mostpop_train1 = model.get_pop_value(train_set1, res)
    # model.save(mostpop_train1, f'./res/{args.dataset}/mostpop/mostpop_train1')

    # save binary-interaction list to csv file
    pred_csv = pd.DataFrame(data=u_binary)
    pred_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_mostpop_train1.csv', index=False)
    
    res_csv = pd.DataFrame(data=u_result)
    res_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_result_mostpop_train1.csv', index=False)


