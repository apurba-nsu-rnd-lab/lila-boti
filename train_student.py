import copy
import random
import os
import shutil
import json
import pickle
from tqdm import tqdm
from prettytable import PrettyTable
import imgaug
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torchsummary import summary
from torchvision import transforms, datasets

from early import EarlyStopping

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from itertools import chain
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from datasets import TestDataset, ApurbaDataset, SynthDataset
from utils_common import preproc_apurba_data, count_parameters, get_padded_labels, decode_prediction, recognition_metrics
from utils_train import preproc_synth_train_data, preproc_synth_valid_data
from model_vgg import get_crnn
import model_vgg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# path = 'temp/resnet18/temp2-bw/ourkd'
# path = 'temp/resnet18/temp2-bw/superkd'
# path = 'temp/resnet18/temp2-bnhtrd/ourkd'
# path = 'temp/resnet18/temp2-bnhtrd/superkd'

# path = 'temp/conv2/temp2-bw/ourkd'
# path = 'temp/conv2/temp2-bw/superkd'
# path = 'temp/conv2/temp2-bnhtrd/ourkd'
# path = 'temp/conv2/temp2-bnhtrd/superkd'


#seeding for reproducability
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#torch.use_deterministic_algorithms(True)
#seeding for reproducability
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
imgaug.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# Path to dataset
# DATA_PATH =  "/home/ec2-user/word_level_ocr/rakib/datasets/banglawriting"
DATA_PATH =  "/home/ec2-user/word_level_ocr/rakib/datasets/bnhtrd"


# Preprocess the data and get grapheme dictionary and labels
# inv_grapheme_dict, words_tr, labels_tr, lengths_tr = preproc_synth_train_data(os.path.join(DATA_PATH, "train_labels.txt"),representation = 'ads')
# words_val, labels_val, lengths_val = preproc_synth_valid_data(os.path.join(DATA_PATH, "valid_labels.txt"), inv_grapheme_dict,representation = 'ads')

# for super!!!
# Preprocess the data and get grapheme dictionary and labels
inv_grapheme_dict, words_tr, labels_tr, lengths_tr = preproc_synth_train_data('/home/ec2-user/word_level_ocr/rakib/datasets/bw_bnhtrd_all/all_labels.txt',representation='ads')
words_tr, labels_tr, lengths_tr = preproc_synth_valid_data(os.path.join(DATA_PATH, "train_labels.txt"), inv_grapheme_dict,representation='ads')
words_val, labels_val, lengths_val = preproc_synth_valid_data(os.path.join(DATA_PATH, "valid_labels.txt"), inv_grapheme_dict, representation='ads')

# Save grapheme dictionary
#https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
with open("./"+path+"/inv_grapheme_dict_synth.pickle", 'wb') as handle:
    pickle.dump(inv_grapheme_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print(inv_grapheme_dict)
#print(len(inv_grapheme_dict))   

##Albumentations noise
data_transform = A.Compose([        
        #A.imgaug.transforms.IAAPerspective (scale=(0.05, 0.05), keep_size=True, always_apply=False, p=0.5),
        #A.imgaug.transforms.IAAAffine (scale=1.0, translate_percent=None, translate_px=None, rotate=5.0, shear=5.0, order=1, cval=0, mode='reflect', always_apply=False, p=0.5),        
        A.ElasticTransform(alpha=0.5, sigma=0, alpha_affine=0, p=0.3),
        A.augmentations.transforms.GaussNoise(var_limit=(120.0, 135.0), mean=0, always_apply=False, p=0.6),
        # A.imgaug.transforms.IAAAdditiveGaussianNoise(loc=1, scale=(2.55, 12.75), per_channel=False, always_apply=False, p=0.4),        
        A.augmentations.transforms.MotionBlur(blur_limit=(3, 6), p=0.3),
        ToTensorV2(),
    ])

# Dataset
train_dataset = SynthDataset(os.path.join(DATA_PATH, "train"), transform=data_transform)
valid_dataset = SynthDataset(os.path.join(DATA_PATH, "valid"), transform=data_transform)

num_train_samples = len(train_dataset)
num_valid_samples = len(valid_dataset)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#teacher model
import torchvision.models as models
import torch.nn as nn

classn = len(inv_grapheme_dict)+1

## TODO: Specify model architecture 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 512)
        self.fc2 = nn.Linear(512, classn)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x

# check if CUDA is available
use_cuda = torch.cuda.is_available()
print(use_cuda)
    
teacher_model = Net()

teacher_model = teacher_model.to(device)

## TODO: Specify model architecture 
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=classn):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#         self.avg_pool = nn.AvgPool2d((4,4), stride=(4,4))

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)  # shape: [batch_size, 64, 32, 32]
#         feature4 = self.avg_pool(out).view(out.size(0), -1)  # self.avg_pool(out): [batch_size, 64, 8, 8]
#         # feature4: [batch_size, 4096]
#         out = self.layer2(out) # shape: [batch_size, 128, 16, 16]
#         feature3 = self.avg_pool(out).view(out.size(0), -1)
#         # feature3: [batch_size, 2048]
#         out = self.layer3(out) # shape: [batch_size, 256, 8, 8]
#         feature2 = F.avg_pool2d(out, out.size(-1))  # [batch_size, channel_num]
#         feature2 = torch.squeeze(feature2)
#         # feature2: [batch_size, 1024]
#         out = self.layer4(out) # shape: [batch_size, 512, 4, 4]
#         #print(out.shape)
#         feature1 = F.avg_pool2d(out, out.size(-1))  # average pooling to [batch_size, channel_num]
#         feature1 = torch.squeeze(feature1)
#         #print(feature1.shape)
#         # out = F.avg_pool2d(out, 4)
#         out = F.adaptive_avg_pool2d(out, 1)
#         out = out.view(out.size(0), -1)
#         #feature1 = out    # [batch_size, 512]
#         out = self.linear(out)
#         # out = F.log_softmax(out, dim=1)

#         return out #, feature1, feature2 #, feature3, feature4

# def resnet18(**kwargs):
#     return ResNet(block=BasicBlock, num_blocks=[2,2,2,2], **kwargs)

# # check if CUDA is available
# use_cuda = torch.cuda.is_available()
# print(use_cuda)
    
# teacher_model = resnet18(num_classes=classn)

import torch.optim as optim

#Resnet18
# for bnhtrd
# teacher_model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/teacher/resnet18_bnhtrd_204.pt'))

# for bw
# teacher_model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/teacher/resnet18_bw_174.pt'))

# for bw_bnhtrd super
# teacher_model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/teacher/resnet18_bw_bnhtrd_216.pt'))

#Conv2
# for bnhtrd
# teacher_model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/teacher/conv2_bnhtrd_204.pt'))

# for bw
# teacher_model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/teacher/conv2_bw_174.pt'))

# for bw_bnhtrd super
teacher_model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/teacher/conv2_bw_bnhtrd_216.pt'))


t = 2
from PIL import Image
import torchvision.transforms as transforms


def load_image(img_path):
    ''' Load in and transform an image'''
    image = Image.open(img_path)
    

     # VGG-16 Takes 224x224 images as input, so we resize all of them and convert data to a normalized torch.FloatTensor    
    in_transform = transforms.Compose([transforms.Resize(32),
                                  transforms.CenterCrop(32),
                                  transforms.Grayscale(num_output_channels=1),
                                  transforms.ToTensor(),
                  
                                  transforms.Normalize(mean=[0.485],
                                  std=[0.229])])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    #print(image.shape)
    return image

dicti = {}
import glob

# #BW minor labels based on support < 500 generated by base model (bwtrain)
# minor_labels = [1, 6, 8, 16, 20, 21, 22, 27, 31, 32, 35, 36, 40, 43, 45, 46, 47, 48, 50, 51, 54, 57, 58, 59, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]

# #BNHTRD minor labels based on support < 500 generated by base model (bnhtrdtrain)
# minor_labels = [1, 11, 26, 30, 34, 43, 48, 54, 56, 57, 58, 59, 61, 62, 63, 64, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196]



for i in range(0,classn):
    model_transfer = teacher_model
    model_transfer.to(device)
    model_transfer.eval()
    batch_pred = []
    k = f'{i:03}'

    
    # bnthrd
    # idd = '/home/ec2-user/word_level_ocr/Ismail/teacher/pre-processed204/train/'+str(k)
    # imgpath = glob.glob('/home/ec2-user/word_level_ocr/Ismail/teacher/pre-processed204/train/'+str(k)+'/*.jpg')

    # #banglawriting
    # idd = '/home/ec2-user/word_level_ocr/Ismail/teacher/pre-processed174/train/'+str(k)
    # imgpath = glob.glob('/home/ec2-user/word_level_ocr/Ismail/teacher/pre-processed174/train/'+str(k)+'/*.jpg')

    # #bw_bnhtrd_super
    idd = '/home/ec2-user/word_level_ocr/Ismail/teacher/pre-processed216/train/'+str(k)
    imgpath = glob.glob('/home/ec2-user/word_level_ocr/Ismail/teacher/pre-processed216/train/'+str(k)+'/*.jpg')

    import random
    at = random.randint(0, 9)
    img= imgpath[at]
    #print(imgpath)
    image = load_image(img)

    if use_cuda:
        image = image.cuda()

    predict = model_transfer(image)
    if i >0:
        predict = F.softmax(predict/t, dim=1)
    else:
        predict = F.log_softmax(predict/t, dim=1)
    pre = predict.data.cpu().argmax()


    if int(pre) != i :
        #print(int(pre))
        for j in range(0,100):
            print(f'{j}')
            img = imgpath[j]
            #print(img)
            image = load_image(img)
            if use_cuda:
                image = image.cuda()
            predict = model_transfer(image)
            predict = F.softmax(predict/t, dim=1)
            pre = predict.data.cpu().argmax()
            if int(pre) == i:
                print(i)
                break



    dicti[k] = predict



def pred_teacher(model_transfer,batch):
    model_transfer.to(device)
    model_transfer.eval()
    batch_pred = []
    ik = 0
    # batch list of words
    for ax in batch:
        #print(ax)
        # remove padded levels
        ax = list(filter(lambda a: a != 0, ax))
        #print(ax)
        #val = len(ax)
        #a=[2,4,6]
        # add skip token
        #ax = list(chain(*[lst[i:i+n] + [ele] if len(lst[i:i+n]) == n else lst[i:i+n] for i in (0, len(lst), n)]))
        #print(ax)
        
        # list of predicted words for a batch without 31 time steps
        predicts = []
        
        # ax list of labels for a word
        for i in ax:
            k = f'{i:03}'
            #idd = 'pre-processed2/valid/'+str(k)
            #imgpath = glob.glob('pre-processed2/valid/'+str(k)+'/*.jpg')
            #imgpath = imgpath[0]
            #print(imgpath)
            #image = load_image(imgpath)
            
            #if use_cuda:
                #image = image.cuda()
           
            predict = dicti[k]
                
            
            #predict = torch.add(predict, 1)
            #predict.to(device)
            predicts.append(predict)
            
            #pre = predict.data.cpu().argmax()
            #print(pre)
            #print(predict.shape)
        
        div = len(ax)
        if(div==0):
            print(div)
        cop = int(31/div)
        add = 31%div

        
        x = torch.cat(predicts)
        #x = x.repeat(cop,1).reshape(div*cop,197)
        npy = x.cpu().detach().numpy()
        arr = np.tile(npy,cop).reshape(div*cop,classn)
        new_row = arr[-1].reshape(1,classn)
        for i in range(add):
            arr = np.concatenate((arr, new_row), axis=0)
        #x = torch.cat((x, x[-1].unsqueeze(dim=0)), 0)
        '''
        if add != 0:
            try:
                


                for i in range(add):
                    arr = np.concatenate((arr, new_row), axis=0)
                    x = torch.cat((x, x[-1].unsqueeze(dim=0)), 0)
            except:
                print(div)
                print(' tarpor' )
                print(cop)'''



        tensors = torch.from_numpy(arr)
        #tensors.to(device)
        # list of word preds with 31 time steps
        batch_pred.append(tensors)
   
    y = torch.cat(batch_pred).reshape(len(batch),31,classn)
    y = y.transpose(0,1)
    y.to(device)
    '''
    np_arr = y.cpu().detach().numpy()
    np_arr = np_arr.reshape(31,len(batch),201)
    y = torch.from_numpy(np_arr)
    y.to(device)'''
    
    return y

def newkldiv(stud,teach):
    
    loss = []
    
    for i in range(31):
        s = torch.nn.functional.log_softmax(stud[i]/t , dim=2)
        t = torch.nn.functional.softmax(teach[i]/t , dim=2)
        kldiv = nn.KLDivLoss(reduction = 'batchmean')(s , t)
        loss.append(kldiv)
    loss = torch.cat(loss).reshape(31)
    loss = loss.mean()
    return loss.cuda
        
#
student_model = model_vgg.get_crnn(len(inv_grapheme_dict)+1,qatconfig=None, bias=True)
student_model = student_model.to(device)
#from 7

#student_model.load_state_dict(torch.load('./bwtrain/init2/init.pth', map_location=device), strict=False)  # ToDO: Turn on strict mode
#student_model.load_state_dict(torch.load('./bwtrain/init2/init.pth', map_location=device))

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import quantization
#Batch Size variable
train_batch_s = 256
valid_batch_s = 256

train_loader = DataLoader(train_dataset, batch_size=train_batch_s, shuffle=True, num_workers=8)
validation_loader = DataLoader(valid_dataset, batch_size=valid_batch_s, shuffle=False, num_workers=8)

###########
for name, param in student_model.named_parameters():
    param.requires_grad = True
###########

criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
criterion = criterion.to(device)
lr = 0.001
optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr, weight_decay=1e-05)

# checkpoint = torch.load('./bnhtrdtrain/init3/init.pth') #for bnhtrd
# checkpoint = torch.load('./bwtrain/init3/init.pth') #for bw
checkpoint = torch.load('./bwtrain/super/init.pth') #for super of bw and bnhtrd
student_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train(epochs=30, lr = 0.0003): #lr=0.001
    
    """
    During Quantization Aware Training, the LR should be 1-10% of the LR used for training without quantization
    """
    
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9, dampening=0, w

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    # new_lr = lr * factor
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2, threshold=0.01, min_lr=0.0001, verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 25], gamma=0.5, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=0.0001)

    #early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(0,100):
        
        student_model.train()

        #necessary variables
        y_true = []
        y_pred = []
        pred_ = []
        label_ = []
        decoded_preds = []
        decoded_labels = []

        total_wer = 0

        print("***Epoch: {}***".format(epoch))
        batch_loss = 0
        #lr = 0.002
        # t = 2
        alpha = 0.5
        for i, (inp, img_names, idx) in enumerate(tqdm(train_loader)):
            
          
            inp = inp.to(device)
            inp = inp.float()/255.
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            img_names = list(img_names)
            words, labels, labels_size = get_padded_labels(idxs, words_tr, labels_tr, lengths_tr)
            
            #print(inp.shape)
            
            teacher_preds = pred_teacher(teacher_model,labels)
            teacher_preds = teacher_preds.cuda()
        
            #teacher_preds = teacher_model(inp)
            #teacher_preds.to(device) 
            #print(teacher_preds.shape)
            #print(teacher_preds[3][0][0])
            # soft_targets = F.log_softmax(teacher_preds /3)
            #print(soft_targets.shape)
            #print(teacher_preds)
         
            z_score = student_model(inp)
         
            preds = torch.nn.functional.log_softmax(z_score , dim=2)
            pr = torch.nn.functional.log_softmax(z_score/t , dim=2)
            #preds = torch.nn.functional.log_softmax(model(inp), dim=2)
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)
            preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long)
            preds_size.to(device)
            #print(preds[3][0])
            #np_arr = preds.cpu().detach().numpy()
            #preds = torch.from_numpy(np_arr)
            
            #print(preds.shape)
            #print(preds[3][0][0])
            # soft_preds = F.log_softmax(preds / t)
            #print(soft_preds.shape)
            
            #kl_div_loss = F.kl_div(soft_preds, soft_targets)
            #kl_div_loss = 
            
            #print(kl_div_loss)
   
            
            #loss = criterion(preds, labels, preds_size, labels_size)
            
            ctc_loss = criterion(preds, labels, preds_size, labels_size)
    
            #loss =  nn.KLDivLoss()(pr , teacher_preds) * ( 1- alpha)  + ctc_loss * alpha
            ty = nn.KLDivLoss()(pr , teacher_preds)
            #ty = ty.cuda()
            loss = ty* (t*t * 2.0 + alpha) + ctc_loss *(1.-alpha)
            #print(loss.item())
            batch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
            labels = labels.detach().numpy()
        
            
            
            for pred, label in zip(preds, labels):
                decoded, _ = decode_prediction(pred, inv_grapheme_dict)
                #_, decoded_label_ = decode_prediction(labels[i], inv_grapheme_dict, gt=True)
                for x, y in zip(decoded, label):
                    
                    y_pred.append(x)
                    
                
                    y_true.append(y)
                    
                _, decoded_pred = decode_prediction(pred, inv_grapheme_dict)
                _, decoded_label = decode_prediction(label, inv_grapheme_dict, gt=True)
                # print("Pred: " + decoded_pred)
                # print("Truth: " + decoded_label)
                
                decoded_preds.append(decoded_pred)
                decoded_labels.append(decoded_label)
            #print(pred_)
                
        scheduler.step()
        
        train_loss = batch_loss/train_batch_s
        print("Epoch Training loss: ", train_loss) #batch_size denominator 32

        print("\n")
        rec_results = recognition_metrics(decoded_preds, decoded_labels, file_name="results.csv")
        print('\n')
        #print(pred_)
        #print(label_)
        print("Absolute Word Match Count: %d" % rec_results['abs_match'])
        print("Word Recognition Rate (WRR): %.4f" % rec_results['wrr'])
        print("Normal Edit Distance (NED): %.4f" % rec_results['total_ned'])
        print("Character Recognition Rate (CRR): %.4f" % rec_results['crr'])
        print("\n")
        print("End of Epoch ",epoch)
        print("\n\n")
        
        with open("./"+path+"/results_training.txt", 'w') as fout:
            for x, y in zip(pred_, label_):
                fout.write("True: {}".format(y))
                fout.write("\n")
                fout.write("Pred: {}".format(x))
                fout.write("\n\n")
    
        metrics = pd.DataFrame([{'epoch': epoch,
                                 'crr': rec_results['crr'],
                                 'wrr': rec_results['wrr'],
                                 'ned': rec_results['total_ned'],
                                 'abs_match': rec_results['abs_match'],
                                 'train_loss': train_loss
                                 }])
        
        metrics.to_csv("./"+path+"/metrics_training.csv", 
                       mode=('w' if epoch==0 else 'a'), index=False, header=(True if epoch==0 else False))
        
        
        #total_wer, _ = compute_wer(pred_, label_)
        #print("Total AED Word Error Rate (Training): %.4f" % total_wer)

        #change in number of labels
        try:    
            report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1), 
                                           zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])
            f1_micro = f1_score(y_true, y_pred, average = 'micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average = 'macro', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
        #change in number of labels


            #Absolute word matching
            #abs_correct = absolute_word_match(pred_, label_)

            #print("Absolute Word Match Count: {}".format(abs_correct))
            #print("Absolute Word Match Percentage: %.4f" % (abs_correct / num_train_samples))
            print("Training Accuracy: %.4f" % accuracy)
            print("Training F1 Micro Score: %.4f" % f1_micro)
            print("Training F1 Macro Score: %.4f" % f1_macro)
            print("\n")
            print("End of Training FOR Epoch {}".format(epoch))
            print("\n\n")

            ##################### Generate Training Report ##############################

            with pd.ExcelWriter("./"+path+"/classification_report_training.xlsx", engine='openpyxl', mode=('w' if epoch==0 else 'a')) as writer:  
                pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch{}'.format(epoch))

            with open("./"+path+"/results_training.txt", 'w') as fout:
                for x, y in zip(pred_, label_):
                    fout.write("True: {}".format(y))
                    fout.write("\n")
                    fout.write("Pred: {}".format(x))
                    fout.write("\n\n")

            metrics = pd.DataFrame([{'epoch': epoch,
                                    'accuracy': accuracy,
                                    'train_loss': train_loss, 

                                    'f1_micro': f1_micro, 
                                    'f1_macro': f1_macro

                                    }])

        except:
            print('error')
        #############################################################################'''
        
        valid_loss = validate(epoch, valid_batch_s, train_loss)
        #early_stopping(valid_loss, student_model)
        
        #if early_stopping.early_stop:
            #print("Early stopping")
            #break
        scheduler.step()
        torch.save(student_model.state_dict(), './'+path+'/epoch{}.pth'.format(epoch))
        
def validate(epoch, valid_batch_s, train_loss):
    
    # evaluate model:
    student_model.eval()

    with torch.no_grad():
        y_true = []
        y_pred = []
        pred_ = []
        label_ = []

        total_wer = 0

        print("***Epoch: {}***".format(epoch))
        batch_loss = 0
        for i, (inp, img_names, idx) in enumerate(tqdm(validation_loader)):

            inp = inp.to(device)
            inp = inp.float()/255.0
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            img_names = list(img_names)
            words, labels, labels_size = get_padded_labels(idxs, words_val, labels_val, lengths_val)
            
            preds = torch.nn.functional.log_softmax(student_model(inp), dim=2)
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)
            preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long)
            preds_size.to(device)

            #validation loss
            loss = criterion(preds, labels, preds_size, labels_size)
            #print(loss)
            batch_loss += loss.item()
            #print(loss.item())

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
            labels = labels.detach().numpy()
            
            for i in range(len(preds)):
                decoded, _ = decode_prediction(preds[i], inv_grapheme_dict)
                for x,y in zip(decoded, labels[i]):
                    y_pred.append(x)
                    
                    y_true.append(y)
                _, decoded_pred_ = decode_prediction(preds[i], inv_grapheme_dict)
                #print(inv_grapheme_dict)
                _, decoded_label_ = decode_prediction(labels[i], inv_grapheme_dict, gt=True)
                #print(decoded_label_)
                
                pred_.append(decoded_pred_)
                label_.append(decoded_label_)

        valid_loss = batch_loss/valid_batch_s
        print("Epoch Validation loss: ", valid_loss) #batch_size denominator 32
        print("\n")
        rec_results = recognition_metrics(pred_, label_, file_name="results.csv")
        print("\n")
        #print(pred_)
        #print(label_)
        print("Absolute Word Match Count: %d" % rec_results['abs_match'])
        print("Word Recognition Rate (WRR): %.4f" % rec_results['wrr'])
        print("Normal Edit Distance (NED): %.4f" % rec_results['total_ned'])
        print("Character Recognition Rate (CRR): %.4f" % rec_results['crr'])
        print("\n")
        print("End of Epoch ",epoch)
        print("\n\n")
        
        with open("./"+path+"/results_validation.txt", 'w') as fout:
            for x, y in zip(pred_, label_):
                fout.write("True: {}".format(y))
                fout.write("\n")
                fout.write("Pred: {}".format(x))
                fout.write("\n\n")
    
        metrics = pd.DataFrame([{'epoch': epoch,
                                 'crr': rec_results['crr'],
                                 'wrr': rec_results['wrr'],
                                 'ned': rec_results['total_ned'],
                                 'abs_match': rec_results['abs_match'],
                                 'train_loss': train_loss
                                 }])
        
        metrics.to_csv("./"+path+"/metrics_validation.csv", 
                       mode=('w' if epoch==0 else 'a'), index=False, header=(True if epoch==0 else False))
        
        #total_wer, _ = compute_wer(pred_, label_)
        #print("Total AED Word Error Rate: %.4f" % total_wer)
        accuracy = accuracy_score(y_true, y_pred)
        print('accuracy ',accuracy)
        
        #change in number of labels
        try:
            report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1), 
                                           zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])
        except:
            print('error')
        f1_micro = f1_score(y_true, y_pred, average = 'micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average = 'macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        

        #Absolute word matching
        #abs_correct = absolute_word_match(pred_, label_)

        #print("Absolute Word Match Count: {}".format(abs_correct))
        #print("Abs Match Percentage: %.4f" % (abs_correct / num_valid_samples))
        print("Accuracy: %.4f" % accuracy)
        print("F1 Micro Score: %.4f" % f1_micro)
        print("F1 Macro Score: %.4f" % f1_macro)
        print("\n")
        print("End of Epoch {}".format(epoch))
        print("\n\n")
        
        ##################### Generate Validation Report ##############################
        
        try:

            with pd.ExcelWriter("./"+path+"/classification_report_validation.xlsx", engine='openpyxl', mode=('w' if epoch==0 else 'a')) as writer:  
                pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch{}'.format(epoch))

            with open("./"+path+"/results_validation.txt", 'w') as fout:
                for x,y in zip(pred_, label_):
                    fout.write("True: {}".format(y))
                    fout.write("\n")
                    fout.write("Pred: {}".format(x))
                    fout.write("\n\n")

            metrics = pd.DataFrame([{'epoch': epoch,
                                    'accuracy': accuracy,
                                    'train_loss': train_loss, 
                                    'valid_loss': valid_loss, 

                                    'f1_micro': f1_micro, 
                                    'f1_macro': f1_macro,

                                    }])

            #metrics.to_csv("./bnhtrdtrain/normalkd1/metrics_validation.csv", 
                           #mode=('w' if epoch==0 else 'a'), index=False, header=(True if epoch==0 else False)) 
        except:
            print('error')
        
        ###############################################################################
        
        return valid_loss
if __name__ == "__main__":
    # execute only if run as a script
    train()
