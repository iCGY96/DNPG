import os
import torch
import argparse
import numpy as np 
import torch.backends.cudnn as cudnn

from dataloader.dataloader import get_dataloaders
from architectures.NetworkPre import FeatureNet
from architectures.GNetworkPre import GFeatureNet
from trainer.FSEval import run_test_fsl
from trainer.GFSEval import run_test_gfsl
import pdb
import logging
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"  #（代表仅使用第0，1号GPU）



model_pool = ['ResNet18','ResNet12','WRN28']
parser = argparse.ArgumentParser('argument for training')

# General Setting
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--num_workers', type=int, default=3, help='num of workers to use')
parser.add_argument('--featype', type=str, default='OpenMeta', choices=['OpenMeta', 'GOpenMeta'], help='type of task: OpenMeta -- FSOR, GOpenMeta --- GFSOR')
parser.add_argument('--restype', type=str, default='ResNet12', choices=model_pool, help='Network Structure')
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet','CIFAR-FS'])
parser.add_argument('--gpus', type=str, default='0')

# Specify folder
parser.add_argument('--logroot', type=str, default='./logs/', help='path to save model')

parser.add_argument('--data_root', type=str, default='/home/zzy/FSL_dataset/miniImageNet', help='path to data root')
parser.add_argument('--pretrained_model_path', type=str, default='/home/zzy/BNPG/DNPG_code/DNPG_check/miniImageNet_phase2_checkpoint.pth', help='path to pretrained model')
#parser.add_argument('--pretrained_model_path', type=str, default='/root/autodl-fs/Tir_fixedFeat_RPL+_max_meta_5shot.pth', help='path to pretrained model') #tier
parser.add_argument('--test_model_path', type=str, default='/home/zzy/BNPG/DNPG_code/log/OpenMeta_miniImageNet/mini_55.pth')
# Meta Setting
parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--n_open_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--n_shots', type=int, default=5, metavar='N', help='Number of shots in test')
parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')
parser.add_argument('--n_train_para', type=int, default=2, metavar='test_batch_size', help='Size of test batch)')
parser.add_argument('--n_train_runs', type=int, default=300, help='Number of training episodes')
parser.add_argument('--n_test_runs', type=int, default=1000, metavar='N', help='Number of test runs')

# Network Flow Path
parser.add_argument('--gamma', type=float, default=2.0, help='loss cofficient for open-mse loss')
parser.add_argument('--tunefeat', type=float, default=0.0, help='update feature parameter')
parser.add_argument('--train_weight_base', action='store_true', help='enable training base class weights')
# Disgarded temporarily
parser.add_argument('--dist_metric', type=str, default='cosine', help='type of negative generator')
parser.add_argument('--comment', default='', type=str)

parser.add_argument('--neg_gen_type', type=str, default='attg', choices=['semang', 'attg', 'att', 'mlp'])
parser.add_argument('--base_seman_calib',type=int, default=1, help='base semantics calibration')
parser.add_argument('--tune_part', type=int, default=2, choices=[1,2])
parser.add_argument('--agg', type=str, default='mlp', choices=['avg', 'mlp'])

parser.add_argument('--held_out', action='store_true')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--open_weight_sum_cali', type=float, default=0.03)

#parser.add_argument('--RPL_loss_temp', type=float, default=0.3)
#parser.add_argument('--RPL_half_flag', type=int, default=0)
#parser.add_argument('--open_weight_sum_cali', type=float, default=0.1)
#parser.add_argument('--bpr_mix_keep_rate', type=float, default=0.6)
#parser.add_argument('--trplet_loss_alpha', type=float, default=0.01)

args = parser.parse_args()



def eval(args, model, meta_test_loader, config):
    params = torch.load(args.test_model_path)['cls_params']
    model.load_state_dict(params, strict=False)
    
    model.eval()
    logging.info('Loaded Model Weight from %s' % args.test_model_path)  

    if args.featype == 'OpenMeta':
        config = run_test_fsl(model, meta_test_loader,config)
        logging.info('Result for %d-shot:' % (args.n_shots))
        for k, v in config.items():
            if k == 'data':
                for k1,v1 in v.items():
                    logging.info('\t\t{}: {}'.format(k1, v1))
            else:
                logging.info('\t{}: {}'.format(k, v))

    else:
        result = run_test_gfsl(model, meta_test_loader)
        logging.info('Result for %d-shot:' % (args.n_shots))
        logging.info('\t Arithmetic Mean: {}'.format(result[0]))
        logging.info('\t Harmonic Mean: {}'.format(result[1]))
        logging.info('\t Delta: {}'.format(result[2]))
        logging.info('\t AUROC: {}'.format(result[3]))
        logging.info('\t F1: {}'.format(result[4]))


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)

    log_file = 'test_%s_%s.log' % (args.comment, args.dataset)
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)

    model_dir = args.test_model_path
    mode = 'openmeta' if args.featype == 'OpenMeta' else 'gopenmeta'
    _, meta_test_loader, n_cls = get_dataloaders(args, mode)

    full_params = torch.load(args.pretrained_model_path)
    params = full_params['feature_params']
    #feat_params = {k: v for k, v in params.items() if 'feature' in k}
    #cls_params = {k: v for k, v in params.items() if 'cls_classifier' in k}
    feat_params = {k.replace('module.','', 1): v for k, v in params.items() if 'feature' in k}
    cls_params = full_params#{k.replace('module.','', 1): v for k, v in params.items() if 'cls_classifier' in k}

    #params = torch.load(args.pretrained_model_path, map_location=torch.device('cpu'))['params']
    #cls_params = {k: v for k, v in params.items() if 'cls_classifier' in k}

    if args.featype == 'OpenMeta':
        model = FeatureNet(args, args.restype, n_cls, (cls_params, meta_test_loader.dataset.vector_array))
    else:
        model = GFeatureNet(args, args.restype, n_cls, (cls_params, meta_test_loader.dataset.vector_array))

    
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True


    ########## Testing Meta-trained Model ##########
    print(args.test_model_path)
    config = {'auroc_type':['prob', 'fscore']}
        
    eval(args, model, meta_test_loader, config)
    logging.info('-----------SEED: %d-----------------' % args.seed)
    logging.info('--------------------------------')

        

