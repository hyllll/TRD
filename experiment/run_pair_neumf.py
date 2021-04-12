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

from daisy.model.pairwise.NeuMFRecommender import PairNeuMF
from daisy.utils.loader import load_rate, split_test, get_ur, PairMFData
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pair-Wise NeuMF recommender test')
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
    parser.add_argument('--sample_method', 
                        type=str, 
                        default='uniform', 
                        help='negative sampling method, options: uniform, item-ascd, item-desc')
    # algo settings
    parser.add_argument('--num_ng', 
                        type=int, 
                        default=4, 
                        help='negative sampling number')
    parser.add_argument('--factor_num', 
                        type=int, 
                        default=32, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=3, 
                        help='number of layers in MLP model')
    parser.add_argument('--model_name', 
                        type=str, 
                        default='NeuMF-end', 
                        help='target model name, if NeuMF-pre plz run MLP and GMF before')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.0, 
                        help='dropout rate')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50, 
                        help='training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=512, 
                        help='batch size for training')
    parser.add_argument('--lamda', 
                        type=float, 
                        default=0.0, 
                        help='regularizer weight')
    parser.add_argument('--out', 
                        default=True, 
                        help='save model or not')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default='BPR', 
                        help='loss function type')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    args = parser.parse_args()


    train_set1 = pd.read_csv(f'../experiment_data/train1_{args.dataset}_{args.prepro}.dat')
    train_set2 = pd.read_csv(f'../experiment_data/train2_{args.dataset}_{args.prepro}.dat')

    test_set = pd.read_csv(f'../experiment_data/test_{args.dataset}_{args.prepro}.dat')

    train_set1['rating'] = 1.0
    train_set2['rating'] = 1.0
    test_set['rating'] = 1.0
    
    # df = pd.concat([train_set, validate_set, test_set], ignore_index=True)
    train_set = pd.concat([train_set1, train_set2], ignore_index=True)
    split_idx = len(train_set)

    df = pd.concat([train_set,  test_set], ignore_index=True)
    df['user'] = pd.Categorical(df['user']).codes
    df['item'] = pd.Categorical(df['item']).codes

    train_set = df.iloc[ : split_idx, :].copy()
    test_set = df.iloc[split_idx: , :].copy()

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()
    print(user_num, item_num)

    # get ground truth
    test_ur = get_ur(test_set)
    # validate_ur = get_ur(validate_set)
    total_train_ur = get_ur(train_set)

    # initial candidate item pool
    item_pool = set(range(item_num))
    candidates_num = args.cand_num

    print('='*50, '\n')
    # retrain model by the whole train set
    # format training data
    train_dataset = PairMFData(train_set, user_num, item_num, args.num_ng, sample_method=args.sample_method)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=4)

    # whether load pre-train model
    model_name = args.model_name
    assert model_name in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']
    GMF_model_path = f'./tmp/{args.dataset}/neumf/GMF.pt'
    MLP_model_path = f'./tmp/{args.dataset}/neumf/MLP.pt'
    NeuMF_model_path = f'./tmp/{args.dataset}/neumf/NeuMF.pt'

    if model_name == 'NeuMF-pre':
        assert os.path.exists(GMF_model_path), 'lack of GMF model'    
        assert os.path.exists(MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(GMF_model_path)
        MLP_model = torch.load(MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    # build recommender model
    model = PairNeuMF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, 
                      args.lr, args.epochs, args.lamda, args.model_name, GMF_model, MLP_model, 
                      args.gpu, args.loss_type)
    model.fit(train_loader)

    if args.out:
        if not os.path.exists(f'./tmp/{args.dataset}/neumf/'):
            os.makedirs(f'./tmp/{args.dataset}/neumf/')
        torch.save(model, f'./tmp/{args.dataset}/neumf/{args.prepro}_neumf.pt')

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
    print('')
    print('Generate recommend list...')
    print('')
    preds = {}

    for u in tqdm(test_ucands.keys()):
        # build a test MF dataset for certain user u to accelerate
        tmp = pd.DataFrame({'user': [u for _ in test_ucands[u]], 
                            'item': test_ucands[u], 
                            'rating': [0. for _ in test_ucands[u]], # fake label, make nonsense
                        })
        tmp_dataset = PairMFData(tmp, user_num, item_num, 0, False)
        tmp_loader = data.DataLoader(tmp_dataset, batch_size=candidates_num, 
                                     shuffle=False, num_workers=0)

        # get top-N list with torch method 
        for items in tmp_loader:
            user_u, item_i = items[0], items[1]
            if torch.cuda.is_available():
                user_u = user_u.cuda()
                item_i = item_i.cuda()
            else:
                user_u = user_u.cpu()
                item_i = item_i.cpu()

            prediction = model.predict(user_u, item_i)
            #_, indices = torch.topk(prediction, args.topk)
            _, indices = torch.topk(prediction, len(test_ucands[u]))
            top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()

        preds[u] = top_n
    
    # convert rank list to binary-interaction
    res = preds.copy()
    u_binary = []
    u_result = []
    record = {}
    u_record = {}
    u_test = []
    i = 0
    for u in preds.keys(): 
        u_record[u] = [u] + res[u].tolist()
        u_result.append(u_record[u])
        preds[u] = [1 if i in test_ur[u] else 0 for i in preds[u]]
        record[u] = [u] + preds[u]
        u_binary.append(record[u])
        u_test.append(u)

    # process topN list and store result for reporting KPI
    print('Save metric@k result to res folder...')
    result_save_path = f'./res/{args.dataset}/neumf/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    
    # save binary-interaction list to csv file
    pred_csv = pd.DataFrame(data=u_binary)
    pred_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_neumf.csv', index=False)

    res_csv = pd.DataFrame(data=u_result)
    res_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_result_neumf.csv', index=False)

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

    res.to_csv(f'{result_save_path}{args.prepro}_neumf.csv', 
               index=False)
    print('='* 20, ' Done ', '='*20)
