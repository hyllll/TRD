# TRD
This is the code for a Top-aware Recommender Distillation framework - TRD with Deep Reinforcement Learning.  The TRD can absorb the essence of state-of-the-art recommenders to further improve the performance of recommendations at the top positions.
## Modules of TRD
For clarify, we use MovieLens-100k dataset as a example and treat the BPRMF method as the teacher model in the TRD framework.
- ### Data preprocess (data_generator.py)
   - Filter the datasets and split the datasets into training set, validation set and test set. 
- ### Training the teacher model (run_pair_mf_train.py)
   - The teacher model can be well trained by using the historical user-item interaction data. After training, we can get the distilled knowledge (i.e., user and item embeddings as well as the basic recommendation list).
- ### Training the student model (run_trd.py)
   - We treat the distilled knowledge as the input and adopt the Deep Q-Networks (DQN) [1] as the student model. The student model aims to reinforce and refine the basic recommendation list.

## Example to run the codes

1. Firstly, we need install the dependent extensions.

   ```python
   python setup.py build_ext --inplace
   ```

2. Then we  run the code to load the dataset and produce the experiment data. If you want to use other datasets, you need modify the code in `data_generator.py`

   ```python
   python data_generator.py
   ```

3. Next,  we run the code to get the distilled knowledge from the teacher model. More details of arguments are available in help message : `python run_pair_mf_train.py --help`
   ```python
   python run_pair_mf_train.py --dataset=ml-100k --prepro=origin
   ```
4. Finally, we train the student model and generate the refined recommendation lists on test set. More details of arguments are available in help message : `python run_trd.py --help`
   ```python
   python run_trd.py --dataset=ml-100k --prepro=origin --method=bprmf --n_actions=20 --pred_score=0
   ```
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
## Acknowledgements

We refer to the following repositories to improve our code:

- state-of-the-art recommendation algorithms  with [daisyRec](https://github.com/AmazingDD/daisyRec) [2]
- DDPG part with [RL_DDPG_Recommendation](https://github.com/bcsrn/RL_DDPG_Recommendation)

## References
[1]  V.  Mnih,  K.  Kavukcuoglu,  D.  Silver,  A.  A.  Rusu,  J.  Veness,  M.  G.Bellemare,  A.  Graves,  M.  Riedmiller,  A.  K.  Fidjeland,  G.  Ostrovski,565et  al.,  [Human-level  control  through  deep  reinforcement learning](),  Nature 518 (7540) (2015) 529-533.

[2] Sun, Zhu and Yu, Di and Fang, Hui and Yang, Jie and Qu, Xinghua and Zhang, Jie and Geng, Cong. [Are we evaluating rigorously? benchmarking recommendation for reproducible evaluation and fair comparison](https://dl.acm.org/doi/abs/10.1145/3383313.3412489). ACM RecSys, 2020.

