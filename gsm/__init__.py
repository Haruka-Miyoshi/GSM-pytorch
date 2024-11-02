import os
import sys
sys.path.append('..')

import torch
from torch import nn
from torch.nn import functional as F
from .model import Model

class GSM(object):
    def __init__(self, x_dim:int, h_dim:int, z_dim:int, lr:float=3e-4, l1_strength:float=1e-3, train_mode:bool=True, save_path="param", model_path="gsm_parameter.path"):
        self.x_dim = x_dim # 入力変数次元数
        self.h_dim = h_dim # 隠れ変数次元数
        self.z_dim = z_dim # 潜在変数次元数

        self.save_path = save_path # パラメータ保存先
        self.model_path = model_path # パラメータファイル名
        self.path = os.path.join(self.save_path, self.model_path) # パス生成
        
        self.mode = train_mode # 学習モード
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # ディバイス設定
        self.model = Model(self.x_dim, self.h_dim, self.z_dim).to(device=self.device) # Model

        if not os.path.exists(save_path):
            os.mkdir(self.save_path)

        if not self.mode:
            # パラメータファイルがない場合における処理を追加
            try:
                self.model.load_state_dict(torch.load(self.path))
            except:
                raise("Not Found model paramter file!!")

        self.lr = lr # 学習係数
        self.l1_strength = torch.tensor(l1_strength, dtype=torch.float32).to(device=self.device) # L1損失における重み係数
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr) # Optimizer
        self.losses = [] # 損失関数

    def tensor(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    def numpy(self, x):
        if self.device == "cpu":
            return x.detach().numpy()
        else:
            return x.detach().cpu().numpy()

    def BCE(self, xh, x):
        loss = F.mse_loss(xh, x, reduction='mean')
        return loss
    
    def KLD(self, mu, logvar):
        loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss
    
    def L1Loss(self, weight):
        return nn.L1Loss()(weight, torch.zeros_like(weight))

    def learn(self, train_loader, epoch:int=10):
        self.model.train() # 学習モード
        
        for e in range(1, epoch+1):
            total_loss = 0.0
            for i, ( batch_data, batch_lbl ) in enumerate(train_loader):
                self.optim.zero_grad()
                x = batch_data.to(device=self.device)
                mu, logvar, z, h, xh = self.model(x)
                loss = self.BCE(xh, x) + self.KLD(mu, logvar) + self.l1_strength * self.L1Loss(self.model.decode.fc.weight)
                loss.backward()
                total_loss += loss.item()
                self.optim.step()
                self.losses.append(total_loss)

            print(f"epoch:{e}, loss:{total_loss}")

        if self.mode:
            torch.save(self.model.state_dict(), self.path)
    
    def encode(self, x):
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(x)
        return mu, self.numpy(mu)
    
    def decode(self, z):
        self.model.eval()
        with torch.no_grad():
            h, xh = self.model.decode(z)
        return h, xh, self.numpy(h), self.numpy(xh)