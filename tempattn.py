import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import math
import pdb
import os
from torch.distributions.multivariate_normal import MultivariateNormal
#from statics import 

class Classifier(nn.Module):
    def __init__(self, args, feat_dim, param_seman, train_weight_base=False):
        super(Classifier, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        self.gen_sam_count = args.gen_sam_count
        self.neiber_choose = args.neiber_choose
        # Weight & Bias for Base
        self.train_weight_base = train_weight_base
        self.init_representation(param_seman)
        if train_weight_base:
            print('Enable training base class weights')
        self.base_info = None
        self.calibrator = SupportCalibrator(nway=args.n_ways, feat_dim=feat_dim, n_head=1, base_seman_calib=args.base_seman_calib, neg_gen_type=args.neg_gen_type)
        self.open_calibrator = SupportCalibrator(nway=1, feat_dim=feat_dim, n_head=1, base_seman_calib=args.base_seman_calib, neg_gen_type='attg')
        self.open_generator = OpenSetGenerater(args.n_ways, feat_dim, n_head=1, neg_gen_type=args.neg_gen_type, agg=args.agg)
        self.metric  = Metric_Cosine()
    def generate_sample_conj(self, supp_features, base_ids):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        alpha = 0.21
        n_gen = 4
        base_mean,base_cov = self.base_info
        base_mean_choose = base_mean[base_ids]
        total_gen = []
              
        for bs in range(len(supp_features)):
            base_cls_mean = base_mean_choose[bs]
            bs_base_cov_choose = base_cov[base_ids][bs]
            feat = supp_features[bs]
            bs_gen = []
            for cls in range(len(feat)):
                bs_cls_mean_list = []
                bs_cls_cov_list = []
                cls_support_feat = feat[cls]
                x_to_base_vector= [cls_support_feat - x for x in base_cls_mean]
                x_to_base_vector = torch.tensor( [item.cpu().detach().numpy() for item in x_to_base_vector] )
                x_to_base_vec_lenth = (x_to_base_vector*x_to_base_vector).sum(1).sqrt()
                x_to_base_att_vector = [x_to_base_vector[x]/x_to_base_vec_lenth[x] for x in range(base_cls_mean.size(0))]
                x_to_base_att_vector = torch.tensor( [item.cpu().detach().numpy() for item in x_to_base_att_vector] )
                a_b_vec = x_to_base_att_vector.unsqueeze(dim=0) + x_to_base_att_vector.unsqueeze(dim=1)
                a_b_vec_lenth = (a_b_vec*a_b_vec).sum(-1)
                find_a_b_vec_lenth = a_b_vec_lenth.view(a_b_vec_lenth.size(0) * a_b_vec_lenth.size(0))
                _,topk_x_y_index = torch.topk(-find_a_b_vec_lenth, self.neiber_choose*2)
                mat_index = []
                for i in topk_x_y_index:
                    mat_index.append([int(i/base_cls_mean.size(0)),int(i%base_cls_mean.size(0))])
                mat_index = [mat_index[i*2] for i in range(self.neiber_choose)]
                pdist = nn.PairwiseDistance(p=2)
                dis_sum = 0
                for i,j in mat_index:
                
                    dis = pdist(base_cls_mean[i].unsqueeze(dim=0),cls_support_feat.unsqueeze(dim=0))
                    dis_sum += dis
                    bs_cls_mean_list.append(base_cls_mean[i] * dis) 
                    bs_cls_cov_list.append(bs_base_cov_choose[i] * dis) 

                    dis = pdist(base_cls_mean[j].unsqueeze(dim=0),cls_support_feat.unsqueeze(dim=0))
                    dis_sum += dis
                    bs_cls_mean_list.append(base_cls_mean[j] * dis) 
                    bs_cls_cov_list.append(bs_base_cov_choose[j] * dis) 

                bs_cls_mean_list = torch.tensor( [item.cpu().detach().numpy() for item in bs_cls_mean_list]).cuda()
                bs_cls_cov_list = torch.tensor( [item.cpu().detach().numpy() for item in bs_cls_cov_list] ).cuda()
                sam_mean = bs_cls_mean_list.sum(0) / (len(bs_cls_mean_list) * dis_sum)
                #new_feat_center = (cls_support_feat + 8*bs_cls_mean_list.mean()/dis_sum)/(12+1)
                new_feat_center = (cls_support_feat + len(bs_cls_mean_list) * sam_mean) / (1+len(bs_cls_mean_list))
                new_feat_cov =  bs_cls_cov_list.sum(0) / (len(bs_cls_cov_list) * dis_sum) + alpha
                gen_model = MultivariateNormal(new_feat_center,new_feat_cov)
                gen = []
                for i in range(self.gen_sam_count):
                    gen.append(gen_model.sample().tolist())

                bs_gen.append(gen)
            total_gen.append(bs_gen)

        return torch.as_tensor(total_gen, dtype=torch.float32).cuda()

    def generate_sample(self, supp_features, base_ids):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        alpha = 0.21
        num_sampled = 550
        base_mean,base_cov = self.base_info
        base_mean_choose = base_mean[base_ids]
        query_feat = supp_features
        cls_protos = base_mean_choose
        scores = self.metric(cls_protos, query_feat)
        _,cls_sim_base_topk = torch.topk(scores, 4, dim=2, largest=True, sorted=True)
        #test = base_ids[cls_sim_base_topk]
        #cls_sim_base_topk_mean = base_mean_choose.unsqueeze(dim=1)
        #cls_sim_base_topk_mean = torch.repeat_interleave(cls_sim_base_topk_mean,5,dim=1)[cls_sim_base_topk,:]
        #new_mean = supp_features + supp_features
        gen_sample = []
        for bs in range(len(supp_features)):
            gen_sam_list = []
            bs_base_mean_choose = base_mean_choose[bs]
            bs_base_cov_choose = base_cov[base_ids][bs]
            bs_cls_sim_base_topk = cls_sim_base_topk[bs]
            bs_qurrey = query_feat[bs]
            for cls in range(len(bs_cls_sim_base_topk)):
                choose_k_base_mean = bs_base_mean_choose[bs_cls_sim_base_topk[cls]]
                choose_k_base_cov = bs_base_cov_choose[bs_cls_sim_base_topk[cls]]
                new_mean = (choose_k_base_mean.sum(0) + bs_qurrey[cls]) / (5)
                new_cov = choose_k_base_cov.sum(0)  / (4) + alpha
                gen_model = MultivariateNormal(new_mean,new_cov,)
            
                gen = []
                for i in range(self.gen_sam_count):
                    gen.append(gen_model.sample().tolist())
                #gen = np.random.multivariate_normal(mean=new_mean.cpu().detach().numpy(), cov=new_cov.cpu().detach().numpy(), size=2)
                #gen = torch.tensor(gen)
                gen_sam_list.append(gen)
            gen_sample.append(gen_sam_list)

        #gen_sample = torch.tensor(gen_sample).cuda()
        #scores = self.metric(gen_sample, query_feat)
        #theta_init_new = torch.as_tensor(theta, dtype=torch.float32, requires_grad=True)

        #A = torch.tensor( [item.cpu().detach().numpy() for item in gen_sample] )

        return torch.as_tensor(gen_sample, dtype=torch.float32).cuda()


        print('test')
        # for bs in range(len(supp_features)):
        #     features = supp_features[bs]
        #     bs_base = base_ids[bs]
        #     #test = torch.index_select(base_mean, 0, bs_base)
        #     base_mean = base_mean[bs_base]
            
    def distribution_calibration(self, query, base_means, base_cov, k,alpha=0.21):
        dist = []
        for i in range(len(base_means)):
            dist.append(np.linalg.norm(query-base_means[i]))
        index = np.argpartition(dist, k)[:k]
        mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
        calibrated_mean = np.mean(mean, axis=0)
        calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

        return calibrated_mean, calibrated_cov

    def generate_sample_fl_mode(self, supp_feature, base_ids):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        base_mean,base_cov = self.base_info
        base_mean_choose = base_mean[base_ids]
        query_feat = supp_feature
        
        #test = base_ids[cls_sim_base_topk]
        #cls_sim_base_topk_mean = base_mean_choose.unsqueeze(dim=1)
        #cls_sim_base_topk_mean = torch.repeat_interleave(cls_sim_base_topk_mean,5,dim=1)[cls_sim_base_topk,:]
        #new_mean = supp_features + supp_features
        gen_sample = []
        for bs in range(len(supp_feature)):
            gen_sam_list = []
            bs_base_mean_choose = base_mean_choose[bs]
            bs_base_cov_choose = base_cov[base_ids][bs]
            bs_qurrey = query_feat[bs]
            for cls in range(len(bs_qurrey)):
                new_mean,new_cov = self.distribution_calibration(bs_qurrey[cls].cpu().detach().numpy(), bs_base_mean_choose.cpu().detach().numpy(), bs_base_cov_choose.cpu().detach().numpy(), 2,alpha=0.21)
                
                gen = np.random.multivariate_normal(mean=new_mean, cov=new_cov, size=79)
                #gen = torch.tensor(gen)
                #gen = np.random.multivariate_normal(mean=new_mean.cpu().detach().numpy(), cov=new_cov.cpu().detach().numpy(), size=2)
                #gen = torch.tensor(gen)
                gen_sam_list.append(gen)
            gen_sample.append(gen_sam_list)

        #gen_sample = torch.tensor(gen_sample).cuda()
        #scores = self.metric(gen_sample, query_feat)
        #theta_init_new = torch.as_tensor(theta, dtype=torch.float32, requires_grad=True)

        #A = torch.tensor( [item.cpu().detach().numpy() for item in gen_sample] )

        return torch.as_tensor(gen_sample, dtype=torch.float32).cuda()


        print('test')
        # for bs in range(len(supp_features)):
        #     features = supp_features[bs]
        #     bs_base = base_ids[bs]
        #     #test = torch.index_select(base_mean, 0, bs_base)
        #     base_mean = base_mean[bs_base]


    def choose_neg_sample(self,raw_features,gen_features):
        scores = self.metric(raw_features, gen_features.view(gen_features.size(0),-1,gen_features.size(-1)))
        scores = scores.view(gen_features.size(0),gen_features.size(1),gen_features.size(2),-1)
        total_chosen = []
        for bs in range(len(scores)):   
            bs_cls_sam_chosen = []      
            for cls in range(len(scores[bs])):
                tem_list = []
                cls_gen_su_scores = scores[bs][cls]
                scores_wo_cls = cls_gen_su_scores[:,torch.arange(cls_gen_su_scores.size(1))!=cls]
                #选取混淆边界样本
                _,scores_wo_cls_max = torch.max(scores_wo_cls,1)
                _,topk_conf_sam = torch.topk(scores_wo_cls_max, 3, dim=0, largest=True, sorted=True)
                chosen_conf_sam = gen_features[bs][cls][topk_conf_sam]


                #选取边缘样本
                scores_to_oth_sum = scores_wo_cls.sum(1)
                _,topk_edge_sam = torch.topk(scores_to_oth_sum, 3, dim=0, largest=True, sorted=True)
                chosen_edge_sam = gen_features[bs][cls][topk_edge_sam]
                #tem_list.extend(chosen_conf_sam.tolist())
                #tem_list.extend(chosen_edge_sam.tolist())
                tem_list = chosen_conf_sam.tolist() + chosen_edge_sam.tolist()
                bs_cls_sam_chosen.append(tem_list)
            
            total_chosen.append(bs_cls_sam_chosen)
        
        return torch.as_tensor(total_chosen, dtype=torch.float32).cuda()


    def forward(self, features, cls_ids, test=False):
        ## bs: features[0].size(0)
        ## support_feat: bs*nway*nshot*D
        ## query_feat: bs*(nway*nquery)*D
        ## base_ids: bs*54
        (support_feat, query_feat, openset_feat) = features

        (nb,nc,ns,ndim),nq = support_feat.size(),query_feat.size(1)
        (supp_ids, base_ids) = cls_ids
        base_weights,base_wgtmem,base_seman,support_seman, base_open_weights = self.get_representation(supp_ids,base_ids)
        support_feat = torch.mean(support_feat, dim=2)
        
        ##############################
        # sampled_data = self.generate_sample_fl_mode(support_feat, base_ids)
        # sopp_data = torch.cat([support_feat.unsqueeze(dim=2), sampled_data], dim=2)
        # supp_protos = torch.mean(sopp_data, dim=2)
        ##############################

        supp_protos,support_attn,attn_used = self.calibrator(support_feat, base_weights, support_seman, base_seman)
        test1 = support_feat.sum(-1)
        test2 = supp_protos.sum(-1)
        test3 = base_open_weights.sum(-1)
        test4 = base_weights.sum(-1)

        sample_loss = 0
        #########新加部分，使用free-lunch模式生成新样本用于neg proto的生成
        #sampled_data = self.generate_sample(supp_protos , base_ids)
        # sampled_data =  self.generate_sample_conj(supp_protos, base_ids)
        
        # for bs in range(len(sampled_data)):
        #     for cls in range(len(sampled_data[bs])):
        #         sample =sampled_data[bs][cls]
        #         #feat = supp_protos[bs][cls]
        #         feat = support_feat[bs][cls]
        #         #cos_sim = F.cosine_similarity(sample, feat)
        #         cos_sim = F.cosine_similarity(sample, feat.unsqueeze(dim=0))
        #         #cos_sim2 = F.cosine_similarity(sample, feat.unsqueeze(dim=0),0)
        #         sample_loss+=cos_sim.sum() / len(cos_sim)
        # sample_loss /= (len(sampled_data) * len(sampled_data[0]))
        
        
        #combine_protos = torch.cat([supp_protos, sampled_data.view(sampled_data.shape[0],sampled_data.shape[1]*sampled_data.shape[2],sampled_data.shape[3])], dim=1)
        #base_weights_t = base_weights.unsqueeze(dim=2)
        #base_weights_t = base_weights_t.repeat(1,1,5,1)
        
        ############## 1对于生成样本进行筛选得到混淆点和边界点用于neg的生成
        #neg_gen_sam = self.choose_neg_sample(supp_protos,sampled_data)

        ############## 2对于生成样本简单和原样本求mean
        # combine_protos = torch.cat([supp_protos.unsqueeze(dim=2), sampled_data], dim=2)
        # combine_protos = torch.mean(combine_protos, dim=2)
        ############## 3对于生成样本每个单独再视作一类
        # combine_protos = torch.cat([supp_protos.unsqueeze(dim=2), sampled_data], dim=2)
        # combine_protos = combine_protos.view(combine_protos.size(0), combine_protos.size(1) * combine_protos.size(2), combine_protos.size(3))

        fakeclass_protos, recip_unit = self.open_generator(supp_protos, base_weights, support_seman=support_seman,base_seman=base_seman,base_open_weights = base_open_weights,support_attn = attn_used)
        test = fakeclass_protos.sum(-1)
        #fakeclass_protos,fakeclass_attn,attn_used = self.open_calibrator(fakeclass_protos, base_open_weights, support_seman, base_seman)


        #########################################################
        #fakeclass_protos, recip_unit = self.open_generator(supp_protos, base_weights, support_seman, base_seman)
        cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
        
        query_cls_scores = self.metric(cls_protos, query_feat)
        openset_cls_scores = self.metric(cls_protos, openset_feat)

        test_cosine_scores = (query_cls_scores,openset_cls_scores)

        query_funit_distance = 1.0- self.metric(recip_unit, query_feat)
        qopen_funit_distance = 1.0- self.metric(recip_unit, openset_feat)
        funit_distance = torch.cat([query_funit_distance,qopen_funit_distance],dim=1)

        return test_cosine_scores, supp_protos, fakeclass_protos, (base_weights,base_wgtmem), funit_distance, 0

    def init_representation(self, param_seman):
        (params,seman_dict) = param_seman
        ######################################
        params_o = torch.load('/root/autodl-fs/random_weight.pth')['params']
        
        #cls_params = {k: v for k, v in params_o.items() if 'open_cls_classifier' in k}

        self.weight_base_open = nn.Parameter(params_o['test.weight'] * 4, requires_grad=self.train_weight_base)
        self.weight_base = nn.Parameter(params['cls_classifier.weight'], requires_grad=self.train_weight_base)
        self.bias_base = nn.Parameter(params['cls_classifier.bias'], requires_grad=self.train_weight_base)
        self.weight_mem = nn.Parameter(params['cls_classifier.weight'].clone(), requires_grad=False)
        test1 = self.weight_base.sum(-1)
        test1 = abs(test1)
        test1 = torch.mean(test1, dim=0)
        test2 = self.weight_base_open.sum(-1)
        test2 = abs(test2)
        test2 = torch.mean(test2, dim=0)
        self.seman = {k:nn.Parameter(torch.from_numpy(v),requires_grad=False).float().cuda() for k,v in seman_dict.items()}
    
    def get_representation(self, cls_ids, base_ids, randpick=False):
        if base_ids is not None:
            base_weights = self.weight_base[base_ids,:]   ## bs*54*D
            base_wgtmem = self.weight_mem[base_ids,:]
            base_seman = self.seman['base'][base_ids,:]
            supp_seman = self.seman['base'][cls_ids,:]
            base_open_weights = self.weight_base_open[base_ids,:] 
            
        else:
            bs = cls_ids.size(0)
            base_weights = self.weight_base.repeat(bs,1,1)
            base_open_weights = self.weight_base_open.repeat(bs,1,1)
            base_wgtmem = self.weight_mem.repeat(bs,1,1)
            base_seman = self.seman['base'].repeat(bs,1,1)
            supp_seman = self.seman['novel_test'][cls_ids,:]
        if randpick:
            num_base = base_weights.shape[1]
            base_size = self.base_size
            idx = np.random.choice(list(range(num_base)), size=base_size, replace=False)
            base_weights = base_weights[:, idx, :]
            base_open_weights = self.weight_base_open[:, idx, :]
            base_seman = base_seman[:, idx, :]
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

        
        if self.neg_gen_type == 'semang':
            support_seman = self._seman_calib(support_seman)
            if self.base_seman_calib:
                base_seman = self._seman_calib(base_seman)

            base_seman = base_seman.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, 300)
            support_seman = support_seman.view(-1, 1, 300)

            base_mem_vis = base_weights
            task_mem_vis = base_weights
            
            base_mem_seman = base_seman
            task_mem_seman = base_seman
            avg_task_mem = torch.mean(torch.cat([task_mem_vis,task_mem_seman],-1), 1, keepdim=True)

            gate_vis = torch.sigmoid(self.task_visfuse(avg_task_mem)) + 1.0
            gate_sem = torch.sigmoid(self.task_semfuse(avg_task_mem)) + 1.0

            base_weights = base_mem_vis * gate_vis 
            base_seman = base_mem_seman * gate_sem

        elif self.neg_gen_type == 'attg':
            base_mem_vis = base_weights
            base_seman = None
            support_seman = None

        elif self.neg_gen_type == 'att':
            base_weights = support_feat
            base_mem_vis = support_feat
            support_seman = None
            base_seman = None

        else:
            return support_feat.view(n_bs,self.nway,-1), None

        support_center, attn_used, support_attn, _ = self.calibrator(support_feat, base_weights, base_mem_vis, support_seman, base_seman)
        test = attn_used.sum(-1)
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
        if neg_gen_type == 'semang':
            self.task_visfuse = nn.Linear(featdim+300,featdim)
            self.task_semfuse = nn.Linear(featdim+300,300)


        self.agg = agg
        if agg == 'mlp':
            self.agg_func = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            
        self.map_sem = nn.Sequential(nn.Linear(300,300),nn.LeakyReLU(0.1),nn.Dropout(0.1),nn.Linear(300,300))

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
        test1 = support_center.sum(-1)
        #self.neg_gen_type='att'
        
        if self.neg_gen_type=='semang':
            support_seman = self._seman_calib(support_seman)
            base_seman = base_seman.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, 300)
            support_seman = support_seman.view(-1, 1, 300)
            
            base_mem_vis = base_weights
            task_mem_vis = base_weights
            
            base_mem_seman = base_seman
            task_mem_seman = base_seman
            avg_task_mem = torch.mean(torch.cat([task_mem_vis,task_mem_seman],-1), 1, keepdim=True)

            gate_vis = torch.sigmoid(self.task_visfuse(avg_task_mem)) + 1.0
            gate_sem = torch.sigmoid(self.task_semfuse(avg_task_mem)) + 1.0
            
            base_weights = base_mem_vis * gate_vis 
            base_seman = base_mem_seman * gate_sem

        
        elif self.neg_gen_type == 'attg':
            base_mem_vis = base_weights
            support_seman = None
            base_seman = None

        elif self.neg_gen_type == 'att':
            base_weights = support_center
            base_mem_vis = support_center
            support_seman = None
            base_seman = None

        else:
            fakeclass_center = support_center.view(bs,-1,640).mean(dim=1, keepdim=True)
            if self.agg == 'mlp':
                fakeclass_center = self.agg_func(fakeclass_center)
            return fakeclass_center, support_center.view(bs, -1, self.featdim)
            #fakeclass_center = support_center.mean(dim=0, keepdim=True)
            #if self.agg == 'mlp':
            #    fakeclass_center = self.agg_func(fakeclass_center)
            #return fakeclass_center, support_center.view(bs, -1, self.featdim)
            

        #output, attcoef, attn_score, value = self.att(support_center, base_weights, base_mem_vis, support_seman, base_seman)  ## bs*nway*nbase
        #test1 = output.view(-1, 1, self.featdim).sum(-1)
        ######### negative_proto加权
        base_mem_vis = base_open_weights
        #test2 = output.sum(-1)
        test3 = base_open_weights.sum(-1)
        output, attcoef, attn_score, value = self.open_att(support_center, base_open_weights, base_mem_vis, support_seman, base_seman,static_attn=support_attn)
        #########
        output = output.view(bs, -1, self.featdim)
        fakeclass_center = output.mean(dim=1,keepdim=True)
        test2 = fakeclass_center.sum(-1)
        
        if self.agg == 'mlp':
            fakeclass_center = self.agg_func(fakeclass_center)
        
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
            test1 = output.sum(-1)
            test2 = residual.sum(-1)
            output = output + residual
            

        test1 = attn.sum(-1)
        test2 = attn_score.sum(-1)
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
        ############################
        #output, attn, attn_score = self.attention(q, k, v, q_sem, k_sem)
        #attn_score = torch.bmm(q, k.transpose(1, 2))
        
        #attn_score /= self.temperature
        attn = static_attn
        #attn = self.attn_dropout(attn)
        test = v.sum(-1)
        test2 = attn.sum(-1)
        output = torch.bmm(attn, v)
        #################################
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if mark_res:
            test1 = output.sum(-1)
            test2 = residual.sum(-1)
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
        test3 = attn.sum(-1)
        attn = self.dropout(attn)

        test = v.sum(-1)
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
    
    
