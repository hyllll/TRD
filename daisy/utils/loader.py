import os
import gc
import json
import random
import pickle
import gzip
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch.utils.data as data

from collections import defaultdict
from sklearn.model_selection import KFold, train_test_split

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def load_rate(src='ml-100k', prepro='origin', binary=True):
    # which dataset will use
    if src == 'ml-100k':
        df = pd.read_csv(f'./data/{src}/u.data', sep='\t', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')

    elif src == 'ml-1m':
        df = pd.read_csv(f'./data/{src}/ratings.dat', sep='::', header=None, 
                        names=['user', 'item', 'rating', 'timestamp'], engine='python')
        # only consider rating >=4 for data density
        df = df.query('rating >= 4').reset_index(drop=True).copy()


    elif src == 'amazon-cloth':
        df = getDF(f'./data/amazon-cloth/reviews_Clothing_Shoes_and_Jewelry_5.json.gz')
        df.drop(columns=['summary', 'helpful', 'reviewTime', 'reviewerName', 'reviewText'], inplace=True)
        df.rename(columns={'asin':'item', 'reviewerID':'user', 'unixReviewTime':'timestamp', 'overall':'rating'}, inplace=True)
        df['rating'] = 1.0
        df['user'] = pd.Categorical(df['user']).codes
        df['item'] = pd.Categorical(df['item']).codes
        user_num = df['user'].nunique()
        item_num = df['item'].nunique()

        return df, user_num, item_num
    else:
        raise ValueError('Invalid Dataset Error')

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rating'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        # encoding user_id and item_id
        df['user'] = pd.Categorical(df['user']).codes
        df['item'] = pd.Categorical(df['item']).codes

        user_num = df['user'].nunique()
        item_num = df['item'].nunique()

        print(f'Finish loading [{src}]-[{prepro}] dataset')
        return df, user_num, item_num

    elif prepro == '5core':
        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        df = df.query('cnt_item >= 5 and cnt_user >= 5').reset_index(drop=True).copy()
        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

        # encoding user_id and item_id
        df['user'] = pd.Categorical(df['user']).codes
        df['item'] = pd.Categorical(df['item']).codes

        user_num = df['user'].nunique()
        item_num = df['item'].nunique()

        print(f'Finish loading [{src}]-[{prepro}] dataset')
        print(user_num, item_num)
        return df, user_num, item_num

    else:
        raise ValueError('Invalid dataset preprocess type, origin/5core expected')

def negative_sampling(user_num, item_num, df, num_ng=4, neg_label_val=0., sample_method='uniform'):
    """
    :param user_num: # of users
    :param item_num: # of items
    :param df: dataframe for sampling
    :param num_ng: # of nagative sampling per sample
    :param neg_label_val: target value for negative samples
    :param sample_method: 'uniform' discrete uniform 
                          'item-desc' descending item popularity, high popularity means high probability to choose
                          'item-ascd' ascending item popularity, low popularity means high probability to choose
    """
    assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
    neg_sample_pool = list(range(item_num))
    if sample_method != 'uniform':
        popularity_item_list = df['item'].value_counts().index.tolist()
        if sample_method == 'item-desc':
            neg_sample_pool = popularity_item_list
        elif sample_method == 'item-ascd':
            neg_sample_pool = popularity_item_list[::-1]

    pair_pos = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for _, row in df.iterrows():
        pair_pos[int(row['user']), int(row['item'])] = 1.0

    neg_df = []
    for _, row in df.iterrows():
        u = int(row['user'])
        i = int(row['item'])
        r = row['rating']
        neg_df.append([u, i, r, 1])
        for _ in range(num_ng):
            if sample_method == 'uniform':
                j = np.random.randint(item_num)
                while (u, j) in pair_pos:
                    j = np.random.randint(item_num)
            # since item-desc, item-ascd both use neg_sample_pool to sample negative item
            elif sample_method in ['item-desc', 'item-ascd']:
                idx = 0
                j = int(neg_sample_pool[idx])
                while (u, j) in pair_pos:
                    idx += 1
                    j = int(neg_sample_pool[idx])

            j = int(j)
            neg_df.append([u, j, neg_label_val, 1])

    neg_df = pd.DataFrame(neg_df, columns=['user', 'item', 'rating', 'timestamp'])
    print('Finish negative sampling......')

    return neg_df

def split_dataset(df):  
    df = df.sort_values(['timestamp']).reset_index(drop=True)
    split_idx = int(np.ceil(len(df) * 0.6))
    split_idx_1 = int(np.ceil(len(df) * 0.7))
    split_idx_2 = int(np.ceil(len(df) * 0.8))
    train_set1, train_set2, validate_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:split_idx_1, :].copy(), df.iloc[split_idx_1:split_idx_2 , :].copy(), df.iloc[split_idx_2: , :].copy()
    return train_set1, train_set2, validate_set, test_set


def split_pandas_data_with_ratios(data, ratios, shuffle=False):
    split_index = np.cumsum(ratios).tolist()[:-1]
    splits = np.split(data, [round(x * len(data)) for x in split_index])
    for i in range(len(ratios)):
        splits[i]["split_index"] = i
    return splits

def get_ur(df):
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))
    return ur

def get_ur_l(df):
    ur = defaultdict(list)
    for _, row in df.iterrows():
        ur[int(row['user'])].append(int(row['item']))
    return ur

def get_model_pred(df):
    data = np.array(df)
    data = data.tolist()
    model_pred = {}
    for l in data:
        u = l.pop(0)
        model_pred[u] = l
    return model_pred

def load_mostpop(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_ir(df):
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir

def build_feat_idx_dict(df:pd.DataFrame, 
                        cat_cols:list=['user', 'item'], 
                        num_cols:list=[]):
    feat_idx_dict = {}
    idx = 0
    for col in cat_cols:
        feat_idx_dict[col] = idx
        idx = idx + df[col].max() + 1
    for col in num_cols:
        feat_idx_dict[col] = idx
        idx += 1
    print('Finish build feature index dictionary......')

    cnt = 0
    for col in cat_cols:
        for _ in df[col].unique():
            cnt += 1
    for col in num_cols:
        cnt += 1
    print(f'Number of features: {cnt}')

    return feat_idx_dict, cnt

class PointMFData(data.Dataset):
    def __init__(self, sampled_df):
        super(PointMFData, self).__init__()
        self.features_fill = []
        self.labels_fill = []
        for _, row in sampled_df.iterrows():
            self.features_fill.append([int(row['user']), int(row['item'])])
            self.labels_fill.append(row['rating'])
        self.labels_fill = np.array(self.labels_fill, dtype=np.float32)

    def __len__(self):
        return len(self.labels_fill)

    def __getitem__(self, idx):
        features = self.features_fill
        labels = self.labels_fill

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]

        return user, item, label

class PointFMData(data.Dataset):
    def __init__(self, sampled_df, feat_idx_dict, 
                 cat_cols, num_cols, loss_type='square_loss'):
        super(PointFMData, self).__init__()

        self.labels = []
        self.features = []
        self.feature_values = []

        assert loss_type in ['square_loss', 'log_loss'], 'Invalid loss type'
        for _, row in sampled_df.iterrows():
            feat, feat_value = [], []
            for col in cat_cols:
                feat.append(feat_idx_dict[col] + row[col])
                feat_value.append(1)
            for col in num_cols:
                feat.append(feat_idx_dict[col])
                feat_value.append(row[col])
            self.features.append(np.array(feat, dtype=np.int64))
            self.feature_values.append(np.array(feat_value, dtype=np.float32))

            if loss_type == 'square_loss':
                self.labels.append(np.float32(row['rating']))
            else: # log_loss
                label = 1 if float(row['rating']) > 0 else 0
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        features = self.features[idx]
        feature_values = self.feature_values[idx]
        return features, feature_values, labels

class PairFMData(data.Dataset):
    def __init__(self, sampled_df, feat_idx_dict, item_num, num_ng, 
                 is_training=True, sample_method='uniform'):
        """
        :param prime sampled_df: dataframe used for sampling
        :param feat_idx_dict: feature index dictionary
        :param item_num: # of item
        :param num_ng: # of negative samples
        :param is_training: whether sampled data used for training
        :param sample_method: 'uniform' discrete uniform 
                              'item-desc' descending item popularity, high popularity means high probability to choose
                              'item-ascd' ascending item popularity, low popularity means high probability to choose
        """
        assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
        neg_sample_pool = list(range(item_num))
        if sample_method != 'uniform':
            popularity_item_list = sampled_df['item'].value_counts().index.tolist()
            if sample_method == 'item-desc':
                neg_sample_pool = popularity_item_list
            elif sample_method == 'item-ascd':
                neg_sample_pool = popularity_item_list[::-1]

        self.features = []
        self.feature_values = []
        self.labels = []

        if is_training:
            pair_pos = set()
            for _, row in sampled_df.iterrows():
                pair_pos.add((int(row['user']), int(row['item'])))
            print('Finish build positive matrix......')

        # construct whole data with negative sampling
        for _, row in sampled_df.iterrows():
            u, i = int(row['user']), int(row['item'])
            if is_training:
                # negative samplings
                for _ in range(num_ng):
                    if sample_method == 'uniform':
                        j = np.random.randint(item_num)
                        while (u, j) in pair_pos:
                            j = np.random.randint(item_num)
                    elif sample_method in ['item-desc', 'item-ascd']:
                        idx = 0
                        j = int(neg_sample_pool[idx])
                        while (u, j) in pair_pos:
                            idx += 1
                            j = int(neg_sample_pool[idx])
                    r = np.float32(1)  # guarantee r_{ui} >_u r_{uj}
                    # TODO if you get a more detail feature dataframe, you need to optimize this part
                    self.features.append([np.array([u + feat_idx_dict['user'], 
                                                    i + feat_idx_dict['item']], dtype=np.int64), 
                                          np.array([u + feat_idx_dict['user'], 
                                                    j + feat_idx_dict['item']], dtype=np.int64)])
                    self.feature_values.append([np.array([1, 1], dtype=np.float32), 
                                                np.array([1, 1], dtype=np.float32)])

                    self.labels.append(np.array(r))
                    
            else:
                j = i
                r = np.float32(1)  # guarantee r_{ui} >_u r_{uj}
                # TODO if you get a more detail feature dataframe, you need to optimize this part
                self.features.append([np.array([u + feat_idx_dict['user'], 
                                                i + feat_idx_dict['item']], dtype=np.int64), 
                                     np.array([u + feat_idx_dict['user'], 
                                               j + feat_idx_dict['item']], dtype=np.int64)])
                self.feature_values.append([np.array([1, 1], dtype=np.float32), 
                                            np.array([1, 1], dtype=np.float32)])
                self.labels.append(np.array(r))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features
        feature_values = self.feature_values
        labels = self.labels

        features_i = features[idx][0]
        features_j = features[idx][1]

        feature_values_i = feature_values[idx][0]
        feature_values_j = feature_values[idx][1]

        labels = labels[idx]

        return features_i, feature_values_i, features_j, feature_values_j, labels


class PairMFData(data.Dataset):
    def __init__(self, sampled_df, user_num, item_num, num_ng, is_training=True, sample_method='uniform'):
        """
        :param sampled_df: prime dataframe used for sampling
        :param user_num: # of user
        :param item_num: # of item
        :param num_ng: # of negative samples
        :param is_training: whether sampled data used for training
        :param sample_method: 'uniform' discrete uniform 
                              'item-desc' descending item popularity, high popularity means high probability to choose
                              'item-ascd' ascending item popularity, low popularity means high probability to choose
        """
        assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
        neg_sample_pool = list(range(item_num))
        if sample_method != 'uniform':
            popularity_item_list = sampled_df['item'].value_counts().index.tolist()
            if sample_method == 'item-desc':
                neg_sample_pool = popularity_item_list
            elif sample_method == 'item-ascd':
                neg_sample_pool = popularity_item_list[::-1]

        super(PairMFData, self).__init__()
        self.is_training = is_training
        self.num_ng = num_ng
        self.sample_num = len(sampled_df)
        self.features_fill = []

        if is_training:
            pair_pos = sp.dok_matrix((user_num, item_num), dtype=np.float32)
            for _, row in sampled_df.iterrows():
                pair_pos[int(row['user']), int(row['item'])] = 1.0
            print('Finish build positive matrix......')

        for _, row in sampled_df.iterrows():
            u, i = int(row['user']), int(row['item'])
            if is_training:
                # negative samplings
                for _ in range(num_ng):
                    if sample_method == 'uniform':
                        j = np.random.randint(item_num)
                        while (u, j) in pair_pos:
                            j = np.random.randint(item_num)
                    elif sample_method in ['item-ascd', 'item-desc']:
                        idx = 0
                        j = int(neg_sample_pool[idx])
                        while (u, j) in pair_pos:
                            idx += 1
                            j = int(neg_sample_pool[idx])
                    j = int(j)
                    r = np.float32(1)  # guarantee r_{ui} >_u r_{uj}
                    self.features_fill.append([u, i, j, r]) 
            else:
                r = np.float32(1)
                self.features_fill.append([u, i, i, r])

        if is_training:
            print(f'Finish negative samplings, sample number is {len(self.features_fill)}......')
    
    def __len__(self):
        return self.num_ng * self.sample_num if self.is_training else self.sample_num

    def __getitem__(self, idx):
        features = self.features_fill
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        label = features[idx][3]

        return user, item_i, item_j, label

""" Item2Vec Specific Process """
class BuildCorpus(object):
    def __init__(self, corpus_df, window=None, max_item_num=20000, unk='<UNK>'):
        """
        Item2Vec Specific Process, building item-corpus by dataframe
        Parameters
        ----------
        corpus_df : pd.DataFrame, the whole dataset
        window : int, window size
        max_item_num : the maximum item pool size,
        unk : str, if there are items beyond existed items, they will all be treated as this value
        """
        # if window is None, means no timestamp, then set max series length as window size
        bad_window = corpus_df.groupby('user')['item'].count().max()
        self.window = bad_window if window is None else window
        self.max_item_num = max_item_num
        self.unk = unk

        # build corpus
        self.corpus = corpus_df.groupby('user')['item'].apply(lambda x: x.values.tolist()).reset_index()

        self.wc = None
        self.idx2item = None
        self.item2idx = None
        self.vocab = None

    def skip_gram(self, record, i):
        iitem = record[i]
        left = record[max(i - self.window, 0): i]
        right = record[i + 1: i + 1 + self.window]
        return iitem, [self.unk for _ in range(self.window - len(left))] + \
                        left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self):
        max_item_num = self.max_item_num
        corpus = self.corpus
        print('building vocab...')
        self.wc = {self.unk: 1}
        for _, row in corpus.iterrows():
            sent = row['item']
            for item in sent:
                self.wc[item] = self.wc.get(item, 0) + 1

        # self.idx2item = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_item_num - 1]
        self.idx2item = sorted(self.wc, key=self.wc.get, reverse=True)[:max_item_num]
        self.item2idx = {self.idx2item[idx]: idx for idx, _ in enumerate(self.idx2item)}
        self.vocab = set([item for item in self.item2idx])
        print('build done')

    def convert(self, corpus_train_df):
        """
        Parameters
        ----------
        corpus_train_df
        Returns
        -------
        dt
        """
        print('converting train by corpus build before...')
        dt = []
        corpus = corpus_train_df.groupby('user')['item'].apply(lambda x: x.values.tolist()).reset_index()
        for _, row in corpus.iterrows():
            sent = []
            for item in row['item']:
                if item in self.vocab:
                    sent.append(item)
                else:
                    sent.append(self.unk)
            for i in range(len(sent)):
                iitem, oitems = self.skip_gram(sent, i)
                dt.append((self.item2idx[iitem], [self.item2idx[oitem] for oitem in oitems]))
        
        print('conversion done')

        return dt


class PermutedSubsampledCorpus(data.Dataset):
    def __init__(self, dt, ws=None):
        if ws is not None:
            self.dt = []
            for iitem, oitems in dt:
                if random.random() > ws[iitem]:
                    self.dt.append((iitem, oitems))
        else:
            self.dt = dt

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx):
        iitem, oitems = self.dt[idx]
        return iitem, np.array(oitems)

""" AE Specific Process """
class AEData(data.Dataset):
    def __init__(self, user_num, item_num, df):
        super(AEData, self).__init__()
        self.user_num = user_num
        self.item_num = item_num

        self.R = np.zeros((user_num, item_num))
        self.mask_R = np.zeros((user_num, item_num))

        for _, row in df.iterrows():
            user, item, rating = int(row['user']), int(row['item']), row['rating']
            self.R[user, item] = float(rating)
            self.mask_R[user, item] = 1.

    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        r = self.R[idx]
        mask_r = self.mask_R[idx]

        return mask_r, r
