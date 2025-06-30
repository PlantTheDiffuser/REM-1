import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F


''' Feature list:
BossExtrude
RevolveBoss
RevolveCut
CutExtrude

'''


def predict_feature() -> str:
    return None

resolution = 150

def ReverseEngineer(img_path):
    # Upper limit for the number of steps to predict features
    max_steps = 10
    featureList = [img_path]
    for i in range(max_steps):
        featureList.append(predict_feature())
    
    return featureList
    

    
    