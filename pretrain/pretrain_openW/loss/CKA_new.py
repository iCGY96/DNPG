from typing import Sequence
from torch import Tensor
import torch
from torch.nn import Module
import numpy as np
import math
import datetime
import time


def centering(k: Tensor, inplace: bool = False) -> Tensor:
    if not inplace:
        k = torch.clone(k)
    means = k.mean(dim=0)
    means -= means.mean() / 2
    k -= means.view(-1, 1)
    k -= means.view(1, -1)

    return k

def kernel_centering(K: Tensor) -> Tensor:
    #start = time.time()
    n = K.size(0)
    unit = torch.ones(n, n).cuda()
    I = torch.eye(n).cuda()
    H = I - unit / n
    #end = time.time()
    #print('centering执行时间：')
    #print(end - start)

    #return torch.mm(torch.mm(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    return torch.mm(H, K)  # KH

# def centering(k: Tensor) -> Tensor:
#     m = k.shape[0]
#     h = torch.eye(m) - torch.ones(m, m) / m
#     return torch.matmul(h, torch.matmul(k, h))
def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X,Y, sigma=None):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_hsic(k: Tensor, l: Tensor, unbiased: bool = False) -> Tensor:
    assert k.shape[0] == l.shape[0], 'Input must have the same size'
    m = k.shape[0]
    if unbiased:
        k.fill_diagonal_(0)
        l.fill_diagonal_(0)
        kl = torch.matmul(k, l)
        score = torch.trace(kl) + k.sum() * l.sum() / ((m - 1) * (m - 2)) - 2 * kl.sum() / (m - 2)
        return score / (m * (m - 3))
    else:
        #k, l = centering(k), centering(l)
        return (k * l).sum() #/ ((m - 1) ** 2)


def cka_score(x1: Tensor, x2: Tensor, gram: bool = False) -> Tensor:
    assert x1.shape[0] == x2.shape[0], 'Input must have the same batch size'
    if not gram:
        x1 = torch.matmul(x1, x1.transpose(0, 1))
        x2 = torch.matmul(x2, x2.transpose(0, 1))
    # cross_score = linear_hsic(x1, x2)
    # self_score1 = linear_hsic(x1, x1)
    # self_score2 = linear_hsic(x2, x2)
    cross_score = linear_hsic(x1, x2)
    self_score1 = linear_hsic(x1, x1)
    self_score2 = linear_hsic(x2, x2)
    return cross_score / torch.sqrt(self_score1 * self_score2)


class CKA_Minibatch(Module):
    """
    Minibatch Centered Kernel Alignment
    Reference: https://arxiv.org/pdf/2010.15327
    """

    def __init__(self):
        super().__init__()
        self.total = 0
        self.cross_hsic, self.self_hsic1, self.self_hsic2 = [], [], []

    def reset(self):
        self.total = 0
        self.cross_hsic, self.self_hsic1, self.self_hsic2 = [], [], []

    def update(self, x1: Tensor, x2: Tensor, gram: bool = False) -> None:
        """
            gram: if true, the method takes gram matrix as input
        """
        assert x1.shape[0] == x2.shape[0], 'Input must have the same batch size'
        self.total += 1
        if not gram:
            x1 = torch.matmul(x1, x1.transpose(0, 1))
            x2 = torch.matmul(x2, x2.transpose(0, 1))
        # start = time.time()
        # test = kernel_HSIC(x1, x2)
        # end = time.time()
        # print('kernel执行时间：')
        # print(end - start)

        # start = time.time()
        # test = linear_hsic(x1, x2)
        # end = time.time()
        # print('linear_hsic执行时间：')
        # print(end - start)

        self.cross_hsic.append(linear_hsic(x1, x2))
        self.self_hsic1.append(linear_hsic(x1, x1))
        self.self_hsic2.append(linear_hsic(x2, x2))

    def compute(self) -> Tensor:
        assert self.total > 0, 'Please call method update(x1, x2) first!'
        cross_score = sum(self.cross_hsic) / self.total
        self_score1 = sum(self.self_hsic1) / self.total
        self_score2 = sum(self.self_hsic2) / self.total
        return cross_score / torch.sqrt(self_score1 * self_score2)


class CKA_Minibatch_Grid(Module):
    '''
    Compute CKA for a 2D grid of features
    '''

    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.cka_loggers = [[CKA_Minibatch() for _ in range(dim2)] for _ in range(dim1)]
        self.dim1 = dim1
        self.dim2 = dim2

    def reset(self):
        for i in range(self.dim1):
            for j in range(self.dim2):
                self.cka_loggers[i][j].reset()

    def update(self, x1: Sequence[Tensor], x2: Sequence[Tensor], gram: bool = False) -> None:
        assert len(x1) == self.dim1, 'Grid dim0 mismatch'
        assert len(x2) == self.dim2, 'Grid dim1 mismatch'
        if not gram:
            x1 = [torch.matmul(x, x.transpose(0, 1)) for x in x1]
            x2 = [torch.matmul(x, x.transpose(0, 1)) for x in x2]
        for i in range(self.dim1):
            for j in range(self.dim2):
                self.cka_loggers[i][j].update(x1[i], x2[j], gram = False)

    def compute(self) -> Tensor:
        result = torch.zeros(self.dim1, self.dim2)
        for i in range(self.dim1):
            for j in range(self.dim2):
                result[i, j] = self.cka_loggers[i][j].compute()
        return result
    
def cka_logits( feat, proto):
    # equivalent to self.linear_CKA, batch computation
    # feat: [b, c, h*w]
    # proto: [num_classes, c, hp*wp]
    def centering(feat):
        assert len(feat.shape) == 3
        return feat - torch.mean(feat, dim=1, keepdims=True)
    
    def cka(va, vb):
        return torch.norm(torch.matmul(va.t(), vb)) ** 2 / (torch.norm(torch.matmul(va.t(), va)) * torch.norm(torch.matmul(vb.t(), vb)))
    
    #if 'centering' in self.args.tag:
    #proto = centering(proto); feat = centering(feat)
    '''
    cka_all = []
    for i in range(b):
        cka_sample = []
        for j in range(num_classes):
            cka_class = cka(feat[i], proto[j])
            cka_sample.append(cka_class)

        cka_all.append(cka_sample)

    logits = torch.tensor(cka_all).cuda()
    print(logits)
    '''
    ### equivalent implementation ###
    proto = proto.unsqueeze(0) # [1, num_classes, c, hp*wp]
    feat = feat.unsqueeze(1) # [b, 1, c, h*w]

    """""" 
    cross_norm = torch.norm(torch.matmul(feat.permute(0, 1, 3, 2), proto), dim=[2,3]) ** 2 # [b, num_classes]
    feat_norm = torch.norm(torch.matmul(feat.permute(0, 1, 3, 2), feat), dim=[2,3]) # [b, 1]
    proto_norm = torch.norm(torch.matmul(proto.permute(0, 1, 3, 2), proto), dim=[2,3]) # [1, num_classes]
    #cross_norm = cross_norm ** 2; feat_norm = feat_norm ** 2; proto_norm = proto_norm ** 2

    logits = cross_norm / (feat_norm * proto_norm) # [b, num_classes]
    #logits = cross_norm * 2 - feat_norm ** 2 - proto_norm ** 2     
    #logits = torch.log(logits)
    """

    power = self.args.aux_param
    if power > 0:
        cross_mat = torch.matmul(feat.permute(0, 1, 3, 2), proto)
        cross_norm = torch.sum(torch.sign(cross_mat) * torch.abs(cross_mat) ** power, dim=[2,3]) # [b, num_classes]

        feat_mat = torch.matmul(feat.permute(0, 1, 3, 2), feat)
        feat_norm = torch.sqrt(torch.sum(torch.sign(feat_mat) * torch.abs(feat_mat) ** power, dim=[2,3])) # [b, 1]

        proto_mat = torch.matmul(proto.permute(0, 1, 3, 2), proto)
        proto_norm = torch.sqrt(torch.sum(torch.sign(proto_mat) * torch.abs(proto_mat) ** power, dim=[2,3])) # [1, num_classes]
    else:
        power = abs(power)
        cross_norm = torch.sum(torch.exp(torch.matmul(feat.permute(0, 1, 3, 2), proto) / power), dim=[2,3]) # [b, num_classes]
        feat_norm = torch.sqrt(torch.sum(torch.exp(torch.matmul(feat.permute(0, 1, 3, 2), feat) / power), dim=[2,3])) # [b, 1]
        proto_norm = torch.sqrt(torch.sum(torch.exp(torch.matmul(proto.permute(0, 1, 3, 2), proto) / power), dim=[2,3])) # [1, num_classes]

    logits = cross_norm / (feat_norm * proto_norm) # [b, num_classes]
    """



    return logits