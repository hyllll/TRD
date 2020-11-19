# TRD
This is the code for a Top-aware Recommender Distillation framework - TRD with Deep Reinforcement Learning.  The TRD can absorb the essence of state-of-the-art recommenders to further improve the performance of recommendations at top positions.

 ## Pre-requisits

### Required environment

- Python 3.6
- Torch (>=1.1.0)
- Numpy (>=1.18.0)
- Pandas (>=0.24.0)

### Datasets

- [MovieLens -100K](https://grouplens.org/datasets/movielens/100k/)

- [MovieLens -1M](https://grouplens.org/datasets/movielens/1m/)

- [Amazon-Cloth](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz)

## Modules of TRD



## Example to run the codes

For clarify, we use MovieLens-100k dataset as a example and treat the BPRMF method as the teacher model in the TRD framework.

1. Firstly, we need install the dependent extensions.

   ```python
   python setup.py build_ext --inplace
   ```

2. Then we  run the code to load the dataset and produce the experiment data. If you want to use other datasets, you can modify the code in `data_generator.py`

   ```python
   python data_generator.py
   ```

3.  Next,  we run the code to get the results of the teacher model.

   ```python
   python run_pair_mf_train.py --dataset=ml-100k --prepro=origin 
   ```

   More details of arguments are available in help message : `python run_pair_mf_train.py --help`

4. Finally, we train the student model and produce the refined recommendation lists on test set.

   ```python
   python run_trd.py --dataset=ml-100k --prepro=origin --method=bprmf --n_actions=20 --pred_score=0
   ```

   More details of arguments are available in help message : `python run_trd.py --help`

## Acknowledgements

We refer to the following repositories to improve our code:

- state-of-the-art recommendation algorithms  with [daisyRec](https://github.com/AmazingDD/daisyRec)
- DDPG part with [RL_DDPG_Recommendation](https://github.com/bcsrn/RL_DDPG_Recommendation)

