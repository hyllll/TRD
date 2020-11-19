import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch.backends.cudnn as cudnn
import torch
import torch.utils.data as data

import sys
sys.path.append('/home/xinghua/Hongyang/Code-submit')
# sys.path.append('/home/workshop/lhy/code-submit')

from daisy.model.Item2VecRecommender import Item2Vec
from daisy.utils.loader import load_rate, split_test, get_ur, BuildCorpus, PermutedSubsampledCorpus
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

def get_weights(wc, idx2item, ss_t, whether_weights):
    wf = np.array([wc[item] for item in idx2item])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2item)
    weights = wf if whether_weights else None

    return vocab_size, weights

def item2vec_data(train_set, test_set, window, item_num, batch_size, ss_t=1e-5, unk='<UNK>', weights=None):
    """
    Parameters
    ----------
    train_set : pd.DataFrame,
    test_set : pd.DataFrame,
    window : int, rolling window size
    item_num : int, the number of total items
    batch_size : batch size
    ss_t : float
    unk : str,
    weights : wheter parse weight
    Returns
    -------
    data_loader: torch.data.Dataset, data generator used for Item2Vec
    vocab_size: int, max item length
    pre.item2idx, dict, the mapping information for item to index code
    """
    df = pd.concat([train_set, test_set], ignore_index=True)
    pre = BuildCorpus(df, window, item_num + 1, unk)
    # pre = BuildCorpus(df, window, item_num, unk)
    pre.build()

    dt = pre.convert(train_set)
    vocab_size, weights = get_weights(pre.wc, pre.idx2item, ss_t, weights)
    data_set = PermutedSubsampledCorpus(dt)  
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True) 

    return data_loader, vocab_size, pre.item2idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='item2Vec recommender test')
    # common settings
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset preprocess op.: origin/5core/10core')
    parser.add_argument('--window', 
                        type=int, 
                        default=2, 
                        help='window size')
    parser.add_argument('--topk', 
                        type=int, 
                        default=50, 
                        help='top number of recommend list')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128, 
                        help='batch_size')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20,
                        help='training epochs')
    parser.add_argument('--factors', 
                        type=int, 
                        default=32, 
                        help='predictive factors numbers in the model')
    parser.add_argument('--cand_num', 
                        type=int, 
                        default=1000, help='No. of candidates item for predict')
    parser.add_argument('--n_negs', 
                        type=int, 
                        default=5, 
                        help='negative sample')
    # algo settings
    parser.add_argument('--out',
                        default=True,
                        help='save model or not')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    parser.add_argument('--use',
                        type=bool, 
                        default=True, 
                        help='whether use gpu')      
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = False

    #temporary used for tuning test result
    train_set1 = pd.read_csv(f'../experiment_data/train1_{args.dataset}_{args.prepro}.dat')
    train_set2 = pd.read_csv(f'../experiment_data/train2_{args.dataset}_{args.prepro}.dat')
    # validate_set = pd.read_csv(f'../experiment_data/validate_{args.dataset}_{args.prepro}_{args.test_method}.dat')
    test_set = pd.read_csv(f'../experiment_data/test_{args.dataset}_{args.prepro}.dat')
    train_set1['rating'] = 1.0
    train_set2['rating'] = 1.0
    # validate_set['rating'] = 1.0
    test_set['rating'] = 1.0
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

    data_loader, vocab_size, item2idx = item2vec_data(train_set, test_set, args.window, item_num, args.batch_size, ss_t=1e-5, unk='<UNK>', weights=None)
    print(vocab_size)
    model = Item2Vec(item2idx, args.dataset, vocab_size, args.factors, args.epochs, args.n_negs, item_num , None, False, True)
    model.fit(data_loader)
    model.build_user_vec(total_train_ur)

    if args.out:
        if not os.path.exists(f'./tmp/{args.dataset}/Item2Vec/'):
            os.makedirs(f'./tmp/{args.dataset}/Item2Vec/')
        torch.save(model, f'./tmp/{args.dataset}/Item2Vec/{args.prepro}_Item2Vec.pt')

    test_ucands = defaultdict(list)

    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
        # sub_item_pool = item_pool - v - total_train_ur[k] -  validate_ur[k] # remove GT & interacted
        sub_item_pool = item_pool - v - total_train_ur[k]
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            samples = random.sample(v, candidates_num)
            test_ucands[k] = list(set(samples))
        else:
            samples = random.sample(sub_item_pool, sample_num)
            test_ucands[k] = list(v | set(samples))

    print('')
    print('Generate recommend list...')
    print('')
    print(len(test_ucands))
    preds = {}
    for u in tqdm(test_ucands.keys()):
        pred_rates = [model.predict(u, i) for i in test_ucands[u]]
        rec_idx = np.argsort(pred_rates)[::-1][:len(test_ucands)]
        top_n = np.array(test_ucands[u])[rec_idx]
        preds[u] = top_n
    
    res = preds.copy()
    u_binary = []
    u_result = []
    record = {}
    u_record = {}
    i = 0
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
    result_save_path = f'./res/{args.dataset}/Item2Vec/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    
    # save binary-interaction list to csv file
    pred_csv = pd.DataFrame(data=u_binary)
    pred_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_Item2Vec.csv', index=False)
    
    res_csv = pd.DataFrame(data=u_result)
    res_csv.to_csv(f'{result_save_path}{args.dataset}_{args.prepro}_result_Item2Vec.csv', index=False)


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

    res.to_csv(f'{result_save_path}{args.prepro}_Item2Vec.csv', 
               index=False)
    print('='* 20, ' Done ', '='*20)