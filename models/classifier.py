import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.layers = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        scores = self.layers(features)
        return scores


class Masker(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle =8192, k = 1024):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       mask = self.bn(self.layers(f))
       z = torch.zeros_like(mask)
       for _ in range(self.k):
           mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
           z = torch.maximum(mask,z)
       return z
    

class Masker_FFT(nn.Module):
    def __init__(self, device,in_dim=2048, num_classes=2048, middle =8192, k = 1024,num_groups=8):
        super(Masker_FFT, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.device = device
        self.num_groups=num_groups

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)
    def forward(self, f):
        # print(f.dtype)
        original_dtype = f.dtype
        f = f.cpu().numpy()
        f_fft = np.fft.fft2(f, axes=(2, 3))
        f_fft_abs = np.abs(f_fft)
        f_fft_abs = np.reshape(f_fft, (64, -1))
        f_fft_pha = np.angle(f_fft)
        f_fft_pha = np.reshape(f_fft, (64, -1))
        f_fft_cmb = np.concatenate((f_fft_abs,f_fft_pha),axis=1)

        f_fft_cmb = (torch.from_numpy(f_fft_cmb)).to(dtype=original_dtype,device=self.device)
        mask = self.bn(self.layers(f_fft_cmb))
        z = torch.zeros_like(mask)
        for _ in range(self.k):
            mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
            z = torch.maximum(mask,z)
        return z[:,:self.num_groups*7*7].to(self.device),z[:,self.num_groups*7*7:].to(self.device)