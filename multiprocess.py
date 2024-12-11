import subprocess as cmd
import sys
from multiprocessing import Process
import time
import os


#time.sleep(3 * 3600)
process_per_gpu = 1

############ cifar ###########

cmd_reg = "python /root/autodl-fs/TANE+oW/train.py --dataset 'miniImageNet' --n_ways 5  --n_shots 5 --featype OpenMeta --learning_rate 0.07  --tunefeat 0.0001 --tune_par 4  --cosine --base_seman_calib 1 --train_weight_base 1 --neg_gen_type attg --gamma %s"
#cmd_reg = "python /home/zzy/FSOR/TANE/train.py --dataset 'miniImageNet' --n_ways 5  --n_shots 5 --featype OpenMeta --learning_rate 0.07  --tunefeat 0.0001 --tune_par 4  --cosine --base_seman_calib 1 --train_weight_base 1 --neg_gen_type att --offset_loss_p %s"

open_weight_sum_cali_list = [0.2,0.1,0.05,0.025,0.075]
RPL_loss_temp_list = [0.2,0.1,0.05,0.025,0.075]
neiber_choose_list = [2, 4, 6, 8, 12]
cut_mix_lam_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
bpr_mix_keep_rate_list = [0.8,0.7,0.6,0.5,0.4,0.3,0.2]
trplet_loss_alpha_list = [0.02,0.03,0.04,0.006]
gamma_list = [0.3,0.5,0.7,1,2,3,5,6]
gpus = [0]
cmd_strs = []
info_strs = []

for p in gamma_list:

    #cmd_str = cmd_reg%(str(p), str(mp), str(mp2), str(ep), str(p), str(mp), str(mp2), str(ep))
    cmd_str = cmd_reg%(str(p))
    cmd_strs.append(cmd_str)
    info_strs.append('gamma %f:'%p)


def process_run(cmd_strs, info_strs, gpu_id):
    for i,cmd_str in enumerate(cmd_strs):
        start_time = time.time()
        cmd_str += ' --gpus %d '%gpu_id
        
        print('executing "%s" ...'%(info_strs[i]))
        text = cmd.getoutput(cmd_str)
        #time.sleep(5)
        print("*******************************************************************")
        print('"%s" ended, time: %s'%(info_strs[i], str(time.time() - start_time)))
        print('"%s" 结果：%s'%(info_strs[i], text))
        print("*******************************************************************")

#gpus = [int(gpu_id) for gpu_id in sys.argv[1].split(',')]
total_gpus = []
for i in range(process_per_gpu):
    total_gpus += gpus
gpus = total_gpus

sub_cmd_strs = []
sub_info_strs = []
for i in range(len(gpus)):
    sub_cmd_strs.append([])
    sub_info_strs.append([])

for i,cmd_str in enumerate(cmd_strs):
    sub_cmd_strs[i%len(gpus)].append(cmd_str)
    sub_info_strs[i%len(gpus)].append(info_strs[i])

for i,gpu in enumerate(gpus):
    p = Process(target=process_run, args=(sub_cmd_strs[i], sub_info_strs[i],gpu))
    p.start()