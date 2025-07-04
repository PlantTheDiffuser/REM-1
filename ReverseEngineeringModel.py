import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F



FeatureList = [
'BossExtrude',
'RevolveBoss',
'RevolveCut',
'CutExtrude',
'Fillet',
'Chamfer',

'Loft?']



def predict_feature() -> str:
    rand = torch.randint(0, len(FeatureList), (1,)).item()
    out = FeatureList[rand]
    rand = 0
    return out

resolution = 150

def ReverseEngineer(img_path):
    # Upper limit for the number of steps to predict features
    max_steps = 10
    featureList = [img_path]
    for i in range(max_steps):
        featureList.append(predict_feature())
    
    return featureList
    

    
    