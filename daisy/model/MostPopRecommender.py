import numpy as np
import pandas as pd
import pickle
import os

class MostPop(object):
    def __init__(self, N=400):
        self.N = N

    def fit(self, train_set):
        '''most popular item'''
        res = train_set['item'].value_counts()
        # self.top_n = res[:self.N].index.tolist()
        self.rank_list = res.index.tolist()[:self.N]

    def predict(self, test_ur, train_ur, topk=10):
        res = {}
        for user in test_ur.keys():
            candidates = self.rank_list
            candidates = [item for item in candidates if item not in train_ur[user]]
            if len(candidates) < topk:
                raise Exception(f'parameter N is too small to get {topk} recommend items')
            res[user] = candidates[:topk]

        return res
    
    def get_pop_value(self, train_set, preds):
        res = train_set['item'].value_counts()
        pop_value = {}
        u_i_pop_value = {}
        for k, v in res.iteritems():
            pop_value[k] = v
        for u in preds.keys():
            pred_u = preds[u]
            for item in pred_u:
                u_i_pop_value[(u, item)] = pop_value[item]
        return u_i_pop_value
    
    def save(self, obj, path):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, path):
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)
    
