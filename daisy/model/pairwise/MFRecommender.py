import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

class PairMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num=32, lamda=0.0,
                 epochs=20, lr=0.01, wd=0.0001, gpuid='0', loss_type='BPR', 
                 early_stop=True, use_cuda=True):
        '''
        user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
        '''
        super(PairMF, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True
        self.use_cuda = use_cuda

        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.lamda = lamda

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        self.loss_type = loss_type
        self.early_stop = early_stop

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        pred_i = (user * item_i).sum(dim=-1)
        pred_j = (user * item_j).sum(dim=-1)

        return pred_i, pred_j

    def fit(self, train_loader):
        if self.use_cuda and torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item_i, item_j, label in pbar:
                if self.use_cuda and torch.cuda.is_available():
                    user = user.cuda()
                    item_i = item_i.cuda()
                    item_j = item_j.cuda()
                    label = label.cuda()
                else:
                    user = user.cpu()
                    item_i = item_i.cpu()
                    item_j = item_j.cpu()
                    label = label.cpu()

                self.zero_grad()
                pred_i, pred_j = self.forward(user, item_i, item_j)

                if self.loss_type == 'BPR':
                    loss = -(pred_i - pred_j).sigmoid().log().sum()
                elif self.loss_type == 'HL':
                    loss = torch.clamp(1 - (pred_i - pred_j) * label, min=0).sum()
                else:
                    raise ValueError(f'Invalid loss type: {self.loss_type}')

                loss += self.lamda * (self.embed_item.weight.norm() + self.embed_user.weight.norm())

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()

            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

    def predict(self, u, i):
        pred_i, _ = self.forward(u, i, i)

        return pred_i.cpu()
