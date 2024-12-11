

# **Learning Unknowns from Unknowns: Diversified Negative Prototypes Generator for Few-Shot Open-Set Recognition**

This is the PyTorch implementation of  [[2408.13373\] Learning Unknowns from Unknowns: Diversified Negative Prototypes Generator for Few-Shot Open-Set Recognition](https://arxiv.org/abs/2408.13373).

Zhenyu Zhang*, Guangyao Chen\*, Yixiong Zou,  Yuhua Li, and Ruixuan Li.
School of Computer Science and Technology, Huazhong University of Science and Technology.
National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University.

the 32nd ACM International Conference on Multimedia (MM ’24).

## 0.dataset and env

see https://github.com/shiyuanh/TANE

## 1.Pretrain
For MiniImageNet:
We provide the pre-trained models for MiniImageNet. Save the pre-trained model to <pretrained_model_path>. or you can train your pretrain model by:

**Phase1:**

backbone pre-training

```python
python ./pretrain/pretrain_model/batch_process.py --gpus 0 --dataset <dataset> --model_path <log_root>  --data_root <data_dir> --batch_size 128 
```

The checkpoints for this stage can be obtained [here](https://drive.google.com/drive/folders/1mj8j5ZChRFLcYMBWEsBBhst8uQTOz_WJ?usp=sharing).

**Phase2:**

The pre-trained model is fixed, and the reverse classification task is trained to obtain open weights.

```python
python ./pretrain/pretrain_openW/batch_process.py --dataset <dataset> --model_path <log_root>  --data_root <data_dir> --batch_size 128 --pret_model_path <pretrain_model_pth_in_1>
```

The checkpoints for this stage can be obtained  [here](https://drive.google.com/drive/folders/1jvJoaG1wwUQEZ6INDGQUurmWksOR87PB?usp=sharing)（***Checkpoints of phase 2 are used for meta-learning.**）

## 2.Meta-learn
```python
python train.py --dataset <dataset> --logroot <log_root>  --data_root <data_dir> \ 
                --n_ways 5 --n_shots 1 --gpus 0 \
                --pretrained_model_path <pretrained_model_path> \
                --learning_rate 0.07 \
                --tunefeat 0.0001 \
                --tune_part 4 \
                --cosine \
                --train_weight_base 1 \ 
```

The checkpoints for this stage can be obtained [here](https://drive.google.com/drive/folders/10OUm0wCg2WZO36FnhXpd3jHyjCtljBWU?usp=sharing)（***Checkpoints of meta-learning are used for testing.**）

## 3.test

```python
python test.py --dataset <dataset>  --data_root <data_dir> \
               --n_ways 5 --n_shots 1 \
               --pretrained_model_path <pretrained_model_path> \
               --featype OpenMeta \
               --test_model_path <test_model_path> \
               --n_test_runs 3000 \
```


​       

## Citation

If you find this repo useful for your research, please consider citing the paper:

```
@inproceedings{zhang2024learning,
  title={Learning Unknowns from Unknowns: Diversified Negative Prototypes Generator for Few-Shot Open-Set Recognition},
  author={Zhang, Zhenyu and Chen, Guangyao and Zou, Yixiong and Li, Yuhua and Li, Ruixuan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={6053--6062},
  year={2024}
}
```

## Acknowledgement

Our code and data are based upon [TANE]([shiyuanh/TANE: Code Repository for "Task-Adaptive Negative Envision for Few-Shot Open-Set Recognition"](https://github.com/shiyuanh/TANE)) and [ARPL]([iCGY96/ARPL: [TPAMI 2022\] Adversarial Reciprocal Points Learning for Open Set Recognition](https://github.com/iCGY96/ARPL)).