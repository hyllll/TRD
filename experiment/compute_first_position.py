import pandas as pd
import numpy as np
import csv

import random
import math
import gc
import argparse

from collections import defaultdict

import sys
# sys.path.append('/home/xinghua/Hongyang/Code-submit')
sys.path.append('/home/workshop/lhy/code-submit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute first position')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='ml-100k', 
                        help='select dataset')
    parser.add_argument('--method', 
                        type=str, 
                        default='bprmf', 
                        help='mostpop, itemknn, bprmf, item2Vec, neumf')
    parser.add_argument('--prepro', 
                        type=str, 
                        default='origin', 
                        help='dataset preprocess op.: origin/5core/10core')
    parser.add_argument('--test_method', 
                        type=str, 
                        default='tfo', 
                        help='method for split test,options: loo/fo/tfo/tloo')
    args = parser.parse_args()
    
    # for method in ['mostpop', 'itemknn', 'bprmf', 'Item2Vec', 'neumf']:
    for method in ['ddpg']:
        df = pd.read_csv(f"./res/{args.dataset}/{method}/{args.dataset}_{args.prepro}_{args.test_method}_{method}.csv")
        df = df.drop(['0'], axis=1)
        data = np.array(df)
        data_list = data.tolist()
        total_first_position = []
    
        for p in data_list:
            position = []
            if 1 in p:
                first_position = p.index(1) + 1
                if first_position > 50:
                    first_position = 100
            else:
                first_position = 100
            total_first_position.append(first_position)
        average_first_position = np.mean(total_first_position)
        print(method, ":the average first position is", average_first_position)
    
        """compute confidence interval"""
        total_num = len(total_first_position)
        boot_means = []
        sample_num = int(total_num * 0.95)
        down = 5
        up = 100 - down
        for _ in range(10000):
            boost_sample = random.sample(total_first_position, sample_num)
            sample_mean = np.array(boost_sample).mean()
            boot_means.append(sample_mean)
        print(method, np.percentile(boot_means, down), np.percentile(boot_means, up))
        # ground_position = 0
        # for i in range(0, len(p)):
        #     if p[i] == 1:
        #         ground_position += i
        # p.reverse()
        # if 1 in p:
        #     end_position = len(p) - p.index(1)
        #     count = pd.value_counts(p)[1]
        #     avg_position =  (ground_position + 1 )/ count
        # else:
        #     end_position = 1000
        #     count = 0
        #     avg_position = 1000


    