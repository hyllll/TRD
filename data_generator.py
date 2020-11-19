import os
import gc
from daisy.utils.loader import load_rate, split_test

# 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'lastfm', 'bx', 'amazon-cloth',
# 'amazon-electronic', 'amazon-book', 'amazon-music', 'epinions', 'yelp', 'citeulike'

# dataset_list = ['amazon-cloth', 'amazon-electronic']
dataset_list = ['ml-100k']

prepro_list = ['origin'] # 'origin', 

if not os.path.exists('./experiment_data/'):
    os.makedirs('./experiment_data/')

for dataset in dataset_list:
    for prepro in prepro_list:
        df, user_num, item_num = load_rate(dataset, prepro, False)
        # print(user_num, item_num)
        train_set1, train_set2, validate_set, test_set = split_dataset(df)
        train_set1.to_csv(f'./experiment_data/train1_{dataset}_{prepro}.dat', index=False)
        train_set2.to_csv(f'./experiment_data/train2_{dataset}_{prepro}.dat', index=False)
        validate_set.to_csv(f'./experiment_data/validate_{dataset}_{prepro}.dat', index=False)
        test_set.to_csv(f'./experiment_data/test_{dataset}_{prepro}.dat', index=False)
        del train_set1, train_set2, validate_set, test_set, df
        print('Finish save train and test set......')

        gc.collect()