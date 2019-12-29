import os
# from tensorboard_logger import configure, log_value
import torch
# from tensorboardX import SummaryWriter
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Bernoulli
from itertools import combinations
from PIL import Image
import time
from torch.utils.data.sampler import Sampler
import random
import math
import torchvision.transforms as transforms
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from hr_net import hr_resnet18, hr_resnet48, hr_resnet64
from inception import inception_v3
import shutil
from functools import wraps
from torchvision import models
from torch.optim.lr_scheduler import MultiStepLR
from efficientnet_pytorch import EfficientNet

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

backbone_fn = {'resnet18': resnet18,
                'resnet34': resnet34,
                'resnet50': resnet50,
                'resnet101': resnet101,
                'resnet152': resnet152,
                'inception_v3': inception_v3,
                'hr_resnet18': hr_resnet18,
                'hr_resnet48': hr_resnet48,
                'hr_resnet64': hr_resnet64}


class Model(nn.Module):
    def __init__(self, backbone='resnet50', class_nums=1010, pretrain_path=''):
        super(Model, self).__init__()
        if 'efficientnet' in backbone:
            self.feature_extractor = EfficientNet.from_pretrained(backbone)
        else:
            self.feature_extractor = backbone_fn[backbone](pretrained=False)
            if pretrain_path:
                print('fan')
                self.feature_extractor.load_state_dict(torch.load(pretrain_path))
        #self.fine_classifier = nn.Linear(1000, fine_label_num)
        self.classifier = nn.Linear(1000, class_nums)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)
        #fine_logits = self.fine_classifier(x)
        #np.savetxt('porn-classification-output2.txt', fine_logits.detach().numpy().reshape(-1), fmt="%.6f")
        coarse_logits = self.classifier(x)

        return coarse_logits

class Model_aux(nn.Module):
    def __init__(self, backbone='resnet50', class_nums=1010, pretrain_path=''):
        super(Model_aux, self).__init__()
        self.feature_extractor = backbone_fn[backbone](pretrained=False)

        if pretrain_path:
            self.feature_extractor.load_state_dict(torch.load(pretrain_path))
        #self.fine_classifier = nn.Linear(1000, fine_label_num)
        self.classifier = nn.Linear(1000, class_nums)
        self.classifier_aux = nn.Linear(1000, class_nums)

    def forward(self, inputs):
        if self.training:
            x, x_aux = self.feature_extractor(inputs)
            logits_aux = self.classifier_aux(x_aux)
            coarse_logits = self.classifier(x)
            return coarse_logits, logits_aux
        else:
            x = self.feature_extractor(inputs)
            coarse_logits = self.classifier(x)
            return coarse_logits
        #fine_logits = self.fine_classifier(x)
        #np.savetxt('porn-classification-output2.txt', fine_logits.detach().numpy().reshape(-1), fmt="%.6f")
        #coarse_logits = self.classifier(x)
        #logits_aux = self.classifier_aux(x_aux)

        #return coarse_logits, logits_aux

class Model_multi(nn.Module):
    def __init__(self, backbone='resnet50', class_nums=1010, pretrain_path=''):
        super(Model_multi, self).__init__()
        self.old_model = Model(backbone=backbone, class_nums=class_nums, pretrain_path=pretrain_path)
        #self.old_model.load_state_dict(torch.load('./models/model_best.pth.tar')['state_dict'])
        self.classifier1 = nn.Linear(1000, 3)
        self.classifier2 = nn.Linear(1000, 4)
        self.classifier3 = nn.Linear(1000, 9)
        self.classifier4 = nn.Linear(1000, 34)
        self.classifier5 = nn.Linear(1000, 57)
        self.classifier6 = nn.Linear(1000, 72)


    def forward(self, inputs):
        x = self.old_model.feature_extractor(inputs)
        #fine_logits = self.fine_classifier(x)
        #np.savetxt('porn-classification-output2.txt', fine_logits.detach().numpy().reshape(-1), fmt="%.6f")
        logits1010 = self.old_model.classifier(x)
        logits3 = self.classifier1(x)
        logits4 = self.classifier2(x)
        logits9 = self.classifier3(x)
        logits34 = self.classifier4(x)
        logits57 = self.classifier5(x)
        logits72 = self.classifier6(x)

        return logits1010, logits3, logits4, logits9, logits34, logits57, logits72
