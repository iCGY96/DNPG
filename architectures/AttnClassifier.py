import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import math
import pdb
import os
from torch.distributions.multivariate_normal import MultivariateNormal
#from statics import
from sklearn import manifold
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self, args, feat_dim, param_seman, train_weight_base=False):
        super(Classifier, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        # Weight & Bias for Base
        self.open_weight_sum_cali = args.open_weight_sum_cali
        self.train_weight_base = train_weight_base
        self.init_representation(param_seman)
        if train_weight_base:
            print('Enable training base class weights')
        self.base_info = None
        self.calibrator = SupportCalibrator(nway=args.n_ways, feat_dim=feat_dim, n_head=1, base_seman_calib=args.base_seman_calib, neg_gen_type=args.neg_gen_type)
        #self.open_calibrator = SupportCalibrator(nway=1, feat_dim=feat_dim, n_head=1, base_seman_calib=args.base_seman_calib, neg_gen_type='attg')
        self.open_generator = OpenSetGenerater(args.n_ways, feat_dim, n_head=1, neg_gen_type=args.neg_gen_type, agg=args.agg)
        self.metric  = Metric_Cosine() #sup protos metric
        self.metric2  = Metric_Cosine() #neg protos metric
        self.metric3  = Metric_Cosine() #recip point metric

        #self.openW_temp = nn.Parameter(torch.tensor(float(1)))
        self.o_bias = nn.Parameter(torch.tensor(float(0)))
        if(args.dataset == 'miniImageNet'):
            self.o_bias.requires_grad = False
    


    def forward(self, features, cls_ids, test=False,mixup_part = None):
        ## bs: features[0].size(0)
        ## support_feat: bs*nway*nshot*D
        ## query_feat: bs*(nway*nquery)*D
        ## base_ids: bs*54
        if(mixup_part != None):
            (aug_data_feat,aug_data_raw_label) = mixup_part
        (support_feat, query_feat, openset_feat) = features
        
        (nb,nc,ns,ndim),nq = support_feat.size(),query_feat.size(1)
        (supp_ids, base_ids) = cls_ids
        base_weights,base_wgtmem,base_seman,support_seman, base_open_weights = self.get_representation(supp_ids,base_ids)
        support_feat = torch.mean(support_feat, dim=2)
        

        supp_protos,support_attn,attn_used = self.calibrator(support_feat, base_weights, support_seman, base_seman)  # prototype calibration
        
        sample_loss = 0

        fakeclass_protos, recip_unit = self.open_generator(supp_protos, base_weights, support_seman=support_seman,base_seman=base_seman,base_open_weights = base_open_weights,support_attn = attn_used) # generate np for each novel class (recip point), and generate task's np.

        cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
        
        
        query_cls_scores_sup = self.metric(supp_protos, query_feat)
        openset_cls_scores_sup = self.metric(supp_protos, openset_feat)
        
        query_cls_scores_neg = self.metric2(fakeclass_protos, query_feat) + self.o_bias
        openset_cls_scores_neg = self.metric2(fakeclass_protos, openset_feat) + self.o_bias

        query_cls_scores = torch.cat([query_cls_scores_sup, query_cls_scores_neg], dim=-1)
        openset_cls_scores = torch.cat([openset_cls_scores_sup, openset_cls_scores_neg], dim=-1)
        
        ########################## Multiple NPs
        test,_1 = torch.max(query_cls_scores[:,:,5:], -1 ) # find the most similar NP
        test = torch.unsqueeze(test, 2)
        query_cls_scores =  torch.cat([query_cls_scores[:,:,:5],test],dim=-1) # cat the most similar NP's logits with prototypes

        test,_2 = torch.max(openset_cls_scores[:,:,5:], -1 ) # find the most similar NP
        test = torch.unsqueeze(test, 2)
        openset_cls_scores = torch.cat([openset_cls_scores[:,:,:5],test],dim=-1) # cat the most similar NP's logits with prototypes

        test_1 = set(_1.view(-1).tolist())
        test_2 = set(_2.view(-1).tolist())
        count_set = test_1 | test_2
        used_size = len(count_set)
        ############################


        test_cosine_scores = (query_cls_scores ,openset_cls_scores )

        query_funit_distance = 1.0- self.metric3(recip_unit, query_feat)
        qopen_funit_distance = 1.0- self.metric3(recip_unit, openset_feat)
        funit_distance = torch.cat([query_funit_distance,qopen_funit_distance],dim=1)

        return test_cosine_scores, supp_protos, fakeclass_protos, (base_weights,base_wgtmem), funit_distance, recip_unit

    def init_representation(self, param_seman):
        (params,seman_dict) = param_seman
        ######################################
        params_RPL = params['RPL_params'] # open weights
        params_GCPL = params['GCPL_params'] # base weights(base class's prototype)
        base_open = params_RPL['centers'].view(-1,4,640).mean(1) # for tier :.view(351,4,640) ; others:.view(64,4,640)
        self.weight_base_open = nn.Parameter( base_open * self.open_weight_sum_cali, requires_grad=self.train_weight_base)
        base = params_GCPL['centers'].view(-1,640)
        self.weight_base = nn.Parameter(base * self.open_weight_sum_cali , requires_grad=self.train_weight_base)
        #self.weight_base =  nn.Parameter(params['feature_params']['cls_classifier.classifier.weight'], requires_grad=self.train_weight_base)
        self.weight_mem = nn.Parameter(base.clone() * self.open_weight_sum_cali , requires_grad=False)


        

    
    def get_representation(self, cls_ids, base_ids, randpick=False):
        if base_ids is not None:
            base_weights = self.weight_base[base_ids,:]   ## bs*54*D
            base_wgtmem = self.weight_mem[base_ids,:]
            base_seman = None
            supp_seman = None
            base_open_weights = self.weight_base_open[base_ids,:] 
            
        else:
            bs = cls_ids.size(0)
            base_weights = self.weight_base.repeat(bs,1,1)
            base_open_weights = self.weight_base_open.repeat(bs,1,1)
            base_wgtmem = self.weight_mem.repeat(bs,1,1)
            base_seman = None
            supp_seman = None
        if randpick:
            num_base = base_weights.shape[1]
            base_size = self.base_size
            idx = np.random.choice(list(range(num_base)), size=base_size, replace=False)
            base_weights = base_weights[:, idx, :]
            base_open_weights = self.weight_base_open[:, idx, :]
            base_seman = None
        

        return base_weights,base_wgtmem,base_seman,supp_seman,base_open_weights


class SupportCalibrator(nn.Module):
    def __init__(self, nway, feat_dim, n_head=1,base_seman_calib=True, neg_gen_type='semang',temp=1):
        super(SupportCalibrator, self).__init__()
        self.nway = nway
        self.feat_dim = feat_dim
        self.base_seman_calib = base_seman_calib

        self.map_sem = nn.Sequential(nn.Linear(300,300),nn.LeakyReLU(0.1),nn.Dropout(0.1),nn.Linear(300,300))

        self.calibrator = MultiHeadAttention(feat_dim//n_head, feat_dim//n_head, (feat_dim,feat_dim),temp = temp)

        self.neg_gen_type = neg_gen_type
        if neg_gen_type == 'semang':
            self.task_visfuse = nn.Linear(feat_dim+300,feat_dim)
            self.task_semfuse = nn.Linear(feat_dim+300,300)

    def _seman_calib(self, seman):
        seman = self.map_sem(seman)
        return seman


    def forward(self, support_feat, base_weights, support_seman, base_seman):
        ## support_feat: bs*nway*640, base_weights: bs*num_base*640, support_seman: bs*nway*300, base_seman:bs*num_base*300        
        n_bs, n_base_cls = base_weights.size()[:2]

        base_weights = base_weights.unsqueeze(dim=1).repeat(1,support_feat.size(1),1,1).view(-1, n_base_cls, self.feat_dim)

        support_feat = support_feat.view(-1,1,self.feat_dim)

        base_mem_vis = base_weights
        base_seman = None
        support_seman = None



        support_center, attn_used, support_attn, _ = self.calibrator(support_feat, base_weights, base_mem_vis, support_seman, base_seman)
        support_center = support_center.view(n_bs,self.nway,-1)
        support_attn = support_attn.view(n_bs,self.nway,-1)
        return support_center, support_attn, attn_used


class OpenSetGenerater(nn.Module):
    def __init__(self, nway, featdim, n_head=1, neg_gen_type='semang', agg='avg'):
        super(OpenSetGenerater, self).__init__()
        self.nway = nway
        self.att = MultiHeadAttention(featdim//n_head, featdim//n_head, (featdim,featdim))
        self.open_att = MultiHeadAttention_static(featdim//n_head, featdim//n_head, (featdim,featdim))
        self.featdim = featdim

        self.neg_gen_type = neg_gen_type

        self.agg = agg
        if agg == 'mlp':#use multiple mlp to generate multiple NP
            self.agg_func1 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            self.agg_func2 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            self.agg_func3 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            self.agg_func4 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            self.agg_func5 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            # self.agg_func6 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim)) # 8NP for cifar
            # self.agg_func7 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim)) # 8NP for cifar
            # self.agg_func8 = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim)) # 8NP for cifar
        #self.map_sem = nn.Sequential(nn.Linear(300,300),nn.LeakyReLU(0.1),nn.Dropout(0.1),nn.Linear(300,300))
        #self.temp_agg_func = nn.Sequential(nn.Linear(featdim * 2,featdim),nn.Dropout(0.2),nn.Linear(featdim,featdim))

    def _seman_calib(self, seman):
        ### feat: bs*d*feat_dim, seman: bs*d*300
        seman = self.map_sem(seman)
        return seman

    def forward(self, support_center, base_weights, support_seman=None, base_seman=None,base_open_weights = None,support_attn=None):
        ## support_center: bs*nway*D
        ## weight_base: bs*nbase*D
        bs = support_center.shape[0]
        n_bs, n_base_cls = base_weights.size()[:2]
        
        base_weights = base_weights.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, self.featdim)
        base_open_weights = base_open_weights.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, self.featdim)
        support_center = support_center.view(-1, 1, self.featdim)
        #self.neg_gen_type='att'
        
        base_mem_vis = base_weights
        support_seman = None
        base_seman = None

        output, attcoef, attn_score, value = self.att(support_center, base_weights, base_mem_vis, support_seman, base_seman)  ## bs*nway*nbase
        ######### negative_proto
        base_mem_vis = base_open_weights


        output, attcoef, attn_score, value = self.open_att(output, base_open_weights, base_mem_vis, support_seman, base_seman,mark_res=False,static_attn=support_attn)
        output = output.view(bs, -1, self.featdim)
        fakeclass_center = output.mean(dim=1,keepdim=True) # mean of novel classes's NPs

        
        if self.agg == 'mlp':#use multiple mlp to generate multiple NP
            fakeclass_center1 = self.agg_func1(fakeclass_center)
            fakeclass_center2 = self.agg_func2(fakeclass_center)
            fakeclass_center3 = self.agg_func3(fakeclass_center)
            fakeclass_center4 = self.agg_func4(fakeclass_center)
            fakeclass_center5 = self.agg_func5(fakeclass_center)
            # fakeclass_center6 = self.agg_func6(fakeclass_center) # 8NP for cifar
            # fakeclass_center7 = self.agg_func7(fakeclass_center) # 8NP for cifar
            # fakeclass_center8 = self.agg_func8(fakeclass_center) # 8NP for cifar
            fakeclass_center = torch.cat((fakeclass_center1, fakeclass_center2, fakeclass_center3, fakeclass_center4, fakeclass_center5), 1)

        
        return fakeclass_center, output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_k, d_v, d_model, n_head=1, dropout=0.1,temp=1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #### Visual feature projection head
        #self.temp = temp
        self.w_qs = nn.Linear(d_model[0], n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model[1], n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model[-1], n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model[0] + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model[1] + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model[-1] + d_v)))

        #### Semantic projection head #######
        self.w_qs_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_ks_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_vs_sem = nn.Linear(300, n_head * d_k, bias=False)
        
        nn.init.normal_(self.w_qs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_ks_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_vs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))            

        self.fc = nn.Linear(n_head * d_v, d_model[0], bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, q_sem=None, k_sem=None, mark_res=True,temp=1):
        ### q: bs*nway*D
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        if q_sem is not None:
            sz_b, len_q, _ = q_sem.size()
            sz_b, len_k, _ = k_sem.size()
            q_sem = self.w_qs_sem(q_sem).view(sz_b, len_q, n_head, d_k)
            k_sem = self.w_ks_sem(k_sem).view(sz_b, len_k, n_head, d_k)
            q_sem = q_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 
            k_sem = k_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) 

        output, attn, attn_score = self.attention(q, k, v, q_sem, k_sem)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if mark_res:
            output = output + residual
            
        return output, attn, attn_score, v


class MultiHeadAttention_static(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_k, d_v, d_model, n_head=1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #### Visual feature projection head
        self.w_qs = nn.Linear(d_model[0], n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model[1], n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model[-1], n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model[0] + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model[1] + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model[-1] + d_v)))

        #### Semantic projection head #######
        self.w_qs_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_ks_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_vs_sem = nn.Linear(300, n_head * d_k, bias=False)
        
        nn.init.normal_(self.w_qs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_ks_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_vs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))

        #self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))            

        self.fc = nn.Linear(n_head * d_v, d_model[0], bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, k, v, q_sem=None, k_sem=None, mark_res=True, static_attn=None):
        ### q: bs*nway*D
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q


        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        if q_sem is not None:
            sz_b, len_q, _ = q_sem.size()
            sz_b, len_k, _ = k_sem.size()
            q_sem = self.w_qs_sem(q_sem).view(sz_b, len_q, n_head, d_k)
            k_sem = self.w_ks_sem(k_sem).view(sz_b, len_k, n_head, d_k)
            q_sem = q_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 
            k_sem = k_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) 
        ############################ use fixed attention weight got in prototype calibration phase
        attn = static_attn
        output = torch.bmm(attn, v)
        #################################
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if mark_res:
            output = output + residual

        return output, attn, attn, v


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, q_sem = None, k_sem = None):

        attn_score = torch.bmm(q, k.transpose(1, 2))
        test1 = self.softmax(attn_score)
        test2 = test1.sum(-1)

        if q_sem is not None:
            attn_sem = torch.bmm(q_sem, k_sem.transpose(1, 2))
            q = q + q_sem
            k = k + k_sem
            attn_score = torch.bmm(q, k.transpose(1, 2))
        
        attn_score /= self.temperature
        attn = self.softmax(attn_score)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn, attn_score
        

class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        supp_center = F.normalize(supp_center, dim=-1) # eps=1e-6 default 1e-12
        query_feature = F.normalize(query_feature, dim=-1)
        logits = torch.bmm(query_feature, supp_center.transpose(1,2))
        return logits * self.temp   
    
    
