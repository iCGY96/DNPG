import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np
import math
import pdb
from architectures.ResNetFeat import create_feature_extractor
from architectures.AttnClassifier import Classifier
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
#from openmixup.models.augments.saliencymix import saliencymix
from tqdm import tqdm

def saliencymix_test(dataloader_trainer):
    with tqdm(dataloader_trainer, total=len(dataloader_trainer), leave=False) as pbar:
            for idx, data in enumerate(pbar):
                  support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label, supp_idx, open_idx = data
                  print('test')
