import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

class PairSLiM(nn.Module):
    def __init__(self, data, user_num, item_num, epochs,
                 lr=0.01, beta=0.0, lamda=0.0, gpuid='0', 
                 loss_type='BPR', early_stop=True):
        super(PairSLiM, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.user_num = user_num
        self.item_num = item_num

        A = np.zeros((user_num, item_num))
        for _, row in data.iterrows():
            user, item = int(row['user']), int(row['item'])
            A[user, item] = 1
        self.A = A  # user-item matrix

        self.W = nn.Embedding(item_num, item_num)
        nn.init.normal_(self.W.weight, std=0.01)
        # force weight >=0 and diag(W) = 0
        self.W.weight.data.clamp_(0)
        self.W.weight.data.copy_(self.W.weight.data * (1 - torch.eye(self.item_num,self.item_num)))

        self.epochs = epochs
        self.lr = lr
        self.beta = beta / 2  # Frobinious regularization
        self.lamda = lamda # lasso regularization

        self.loss_type = loss_type
        self.early_stop = early_stop

    def forward(self, user, item_i, item_j):
        tensor_A = torch.from_numpy(self.A).to(torch.float32)
        if torch.cuda.is_available():
            tensor_A = tensor_A.cuda()
        else:
            tensor_A = tensor_A.cpu()

        ru = tensor_A[user]
        wi = self.W(item_i)
        wj = self.W(item_j)

        pred_i = (ru * wi).sum(dim=-1)
        pred_j = (ru * wj).sum(dim=-1)

        return pred_i, pred_j

    def fit(self, train_loader):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()
        
        optimizer = optim.SGD(self.parameters(), self.lr)
        
        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, item_i, item_j, label in pbar:
                if torch.cuda.is_available():
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

                loss += self.beta * self.W.weight.norm() + self.lamda * self.W.weight.norm(p=1)

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()
            
            self.eval()

            # reset W element >= 0 and diag(W) = 0
            self.W.weight.data.clamp_(0)
            if torch.cuda.is_available():
                tmp_eye = torch.eye(self.item_num, self.item_num).cuda()
            else:
                tmp_eye = torch.eye(self.item_num, self.item_num).cpu()
            self.W.weight.data.copy_(self.W.weight.data * (1 - tmp_eye))

            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

        # get prediction rating matrix
        self.A_tilde = self.A.dot(self.W.weight.data.cpu().numpy())

    def predict(self, u, i):
        return self.A_tilde[u, i]