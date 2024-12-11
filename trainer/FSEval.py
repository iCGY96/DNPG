from __future__ import print_function

import sys, os, pdb 
import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.metrics import f1_score
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import manifold

###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#导入iris数据
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier 
import torchvision.transforms as transforms
###
def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
    std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
    transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])

    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
    	img_tensor = img_tensor.numpy()
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        
    return img


def run_test_fsl(net, openloader,config,encoder=None, generator=None, n_ways=5,n_shots=1,scale=4):
    net = net.eval()
    auroc_type = config['auroc_type']
    
    with tqdm(openloader, total=len(openloader), leave=False) as pbar:
        acc_trace = []
        auroc_trace = {k:[] for k in auroc_type}
        count_list = []
        for idx, data in enumerate(pbar):
            feats, labels, probs = compute_feats(net, data)
            acc, auroc = eval_fsl_nplus1(feats, labels, probs, auroc_type)

            acc_trace.append(acc)
            for t in auroc_type:
                if auroc[t] is None:
                    continue
                auroc_trace[t].append(auroc[t])

            pbar.set_postfix({
                    "OpenSet MetaEval Acc":'{0:.2f}'.format(acc),
                    "AUROC-%s MetaEval:" % auroc_type[0]:'{0:.2f}'.format(auroc[auroc_type[0]])
                })

        config['data'] = {'acc': mean_confidence_interval(acc_trace)}
        for t in auroc_type:
            config['data']['auroc_%s'%t] = mean_confidence_interval(auroc_trace[t])

    return config
                        


def eval_fsl_nplus1(feats, labels, probs, auroc_type=['prob',]):
    cls_protos,query_feats,open_feats = feats
    supp_label, query_label, open_label = labels
    num_query = query_label.shape[0]
    supp_label = supp_label.view()
    all_probs = np.concatenate(probs, axis=0)
    
    auroc = dict()

    if 'prob' in auroc_type:
        auroc_score = all_probs[:,-1]
        
        auroc_result = metrics.roc_auc_score(1-open_label,auroc_score)
        auroc['prob'] = auroc_result

    if 'fscore' in auroc_type:
        num_open = len(open_label) - len(query_label)
        num_way = 5
        all_labels = np.concatenate([query_label, num_way * np.ones(num_open)], -1).astype(int)
        ypred = np.argmax(all_probs, axis=-1)
        auroc['fscore'] = f1_score(all_labels, ypred, average='macro', labels=np.unique(ypred))

    #assert all_probs.shape[-1] == 6
    num_query = query_label.shape[0]
    query_pred = np.argmax(all_probs[:num_query,:-1], axis=-1)
    acc = metrics.accuracy_score(query_label, query_pred)
    
    return acc, auroc



def compute_feats(net, data):
    with torch.no_grad():
        # Data Preparation
        support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label, supp_idx, open_idx = data

        # Data Conversion & Packaging
        support_data,support_label              = support_data.float().cuda(),support_label.cuda().long()
        query_data,query_label                  = query_data.float().cuda(),query_label.cuda().long()
        suppopen_data,suppopen_label            = suppopen_data.float().cuda(),suppopen_label.cuda().long()
        openset_data,openset_label              = openset_data.float().cuda(),openset_label.cuda().long()
        supp_idx, open_idx= supp_idx.long(), open_idx.long()
        openset_label = net.n_ways * torch.ones_like(openset_label)
        the_img     = (support_data, query_data, suppopen_data, openset_data)
        the_label   = (support_label,query_label,suppopen_label,openset_label)
        the_conj    = (supp_idx, open_idx)

        # Tensor Input Preparation
        features, cls_protos, cosine_probs= net(the_img,the_label,the_conj,test=True)

        ###############################################count FLOAPs
        # from thop import profile
        # flops, params = profile(net, (the_img,the_label,the_conj,True,))
        # print('flops: ', flops, 'params: ', params)
        ################################################
        (supp_feat, query_feat, openset_feat) = features

        # Numpy Input Preparation
        cls_protos_numpy = F.normalize(cls_protos.view(-1,net.feat_dim),p=2,dim=-1).cpu().numpy()
        supplabel_numpy = support_label.view(supp_feat.shape[1:-1]).cpu().numpy()
        querylabel_numpy = query_label.view(-1).cpu().numpy()
        supp_feat_numpy = F.normalize(supp_feat[0].view(-1,net.feat_dim),p=2,dim=-1).cpu().numpy()
        queryfeat_numpy = F.normalize(query_feat[0],p=2,dim=-1).cpu().numpy()
        openfeat_numpy = F.normalize(openset_feat[0],p=2,dim=-1).cpu().numpy()
        open_label = np.concatenate((np.ones(query_label.size(1)),np.zeros(openset_label.size(1))))

        # Numpy Probs Preparation
        query_cls_probs, openset_cls_probs = cosine_probs
        query_cls_probs = query_cls_probs[0].cpu().numpy()
        openset_cls_probs = openset_cls_probs[0].cpu().numpy()
        cosine_probs = (query_cls_probs, openset_cls_probs)
                
    return (cls_protos_numpy,queryfeat_numpy,openfeat_numpy), (supplabel_numpy, querylabel_numpy, open_label), cosine_probs



def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h

