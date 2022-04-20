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

import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from datasets import TestDataset, ApurbaDataset, SynthDataset
from utils_common2 import preproc_apurba_data, count_parameters, get_padded_labels, decode_prediction, recognition_metrics
from utils_train import preproc_synth_train_data, preproc_synth_valid_data
from model_vgg import get_crnn
import model_vgg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# path = 'temp/resnet18/temp2-bnhtrd/normalkd'
path = 'temp/resnet18/temp2-bw/normalkd'


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
DATA_PATH =  "/home/ec2-user/word_level_ocr/rakib/datasets/banglawriting"
# DATA_PATH =  "/home/ec2-user/word_level_ocr/rakib/datasets/bnhtrd"


# Preprocess the data and get grapheme dictionary and labels
# inv_grapheme_dict, words_tr, labels_tr, lengths_tr = preproc_synth_train_data('/home/ec2-user/word_level_ocr/rakib/datasets/bw_bnhtrd_all/all_labels.txt',representation='ads')
# words_tr, labels_tr, lengths_tr = preproc_synth_valid_data(os.path.join(DATA_PATH, "train_labels.txt"), inv_grapheme_dict,representation='ads')
# words_val, labels_val, lengths_val = preproc_synth_valid_data(os.path.join(DATA_PATH, "valid_labels.txt"), inv_grapheme_dict, representation='ads')

# Preprocess the data and get grapheme dictionary and labels
inv_grapheme_dict, words_tr, labels_tr, lengths_tr = preproc_synth_train_data(os.path.join(DATA_PATH, "train_labels.txt"),representation='ads')
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

classn = len(inv_grapheme_dict)
## TODO: Specify model architecture 

#
student_model = model_vgg.get_crnn(len(inv_grapheme_dict)+1,qatconfig=None)
student_model = student_model.to(device)
#from 7
#student_model.load_state_dict(torch.load('./bwtrain/init2/init.pth', map_location=device))
teacher_model2 = model_vgg.get_crnn(len(inv_grapheme_dict)+1,qatconfig=None)
teacher_model2 = teacher_model2.to(device)
# for bnhtrd
# teacher_model2.load_state_dict(torch.load('./bnhtrdtrain/init3/epoch97.pth', map_location=device))
# for bw
teacher_model2.load_state_dict(torch.load('./bwtrain/init3/epoch95.pth', map_location=device))

teacher_model2.eval()
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

# checkpoint = torch.load('./bnhtrdtrain/init3/init.pth') # bnhtrd
checkpoint = torch.load('./bwtrain/init3/init.pth') # bw
student_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train(epochs=30, lr = 0.0003): #lr=0.001
    
    """
    During Quantization Aware Training, the LR should be 1-10% of the LR used for training without quantization
    """
    
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9, dampening=0, weight_decay=1e-05)
    
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

        total_wer = 0

        print("***Epoch: {}***".format(epoch))
        batch_loss = 0
        # lr = 0.002
        t = 2
        alpha = 0.5
        for i, (inp, img_names, idx) in enumerate(tqdm(train_loader)):
        
            inp = inp.to(device)
            inp = inp.float()/255.0
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            img_names = list(img_names)
            words, labels, labels_size = get_padded_labels(idxs, words_tr, labels_tr, lengths_tr)
            #print(words)
            #print(labels)
            #print(inp.shape)
            
            #teacher_preds = pred_teacher(teacher_model,labels)
            #teacher_preds = teacher_preds.cuda()
            #teacher_preds = teacher_model(inp)
            #teacher_preds.to(device) 
            #print(teacher_preds.shape)
            #print(teacher_preds[3][0][0])
            #soft_targets = F.log_softmax(teacher_preds /3)
            #print(soft_targets.shape)
            #print(teacher_preds)
            z_score = student_model(inp)
         
            preds = torch.nn.functional.log_softmax( z_score, dim=2)
            pr = torch.nn.functional.log_softmax(z_score/t , dim=2)
            #preds = torch.nn.functional.log_softmax(model(inp), dim=2)
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)
            preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long)
            preds_size.to(device)
            
            #teacher model 2
            
            preds2 = torch.nn.functional.softmax(teacher_model2(inp)/t , dim=2)
            #preds2 = preds2.cuda()
            kl = preds2.cpu().data.numpy()
            kl = torch. from_numpy(kl)
            kl = kl.cuda()
            #div = F.softmax(kl/t)
            #div = div.cuda()
            #loss = criterion(preds, labels, preds_size, labels_size)'''
            
            ctc_loss = criterion(preds, labels, preds_size, labels_size)
            #(nn.KLDivLoss()(F.log_softmax(preds / t), F.softmax(teacher_preds/t))+nn.KLDivLoss()(F.log_softmax(preds / t), F.softmax(preds2/t)) ) * (t*t * 2.0 + alpha)  + +nn.KLDivLoss()(F.log_softmax(preds / t), F.softmax(preds2/t))
            ty = nn.KLDivLoss()(pr,kl)
            
            loss = ty* (t*t * 2.0 + alpha) + ctc_loss *(1.-alpha)
            #print(loss.item())
            batch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
            labels = labels.detach().numpy()
            
            for i in range(len(preds)):
                decoded, _ = decode_prediction(preds[i], inv_grapheme_dict)
                for x,y in zip(decoded, labels[i]):
                    y_pred.append(x)
                    #print(x)
                    #print('t')
                    #print(y)
                    y_true.append(y)
                _, decoded_pred_ = decode_prediction(preds[i], inv_grapheme_dict)
                #print(inv_grapheme_dict)
                _, decoded_label_ = decode_prediction(labels[i], inv_grapheme_dict, gt=True)
                #print(decoded_label_)
                
                pred_.append(decoded_pred_)
            
                label_.append(decoded_label_)
            #print(pred_)
                
        scheduler.step()
        
        train_loss = batch_loss/train_batch_s
        print("Epoch Training loss: ", train_loss) #batch_size denominator 32

        print("\n")
        rec_results = recognition_metrics(pred_, label_, file_name="results.csv")
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
        
        #scheduler.step()
        torch.save(student_model.state_dict(), './'+path+'/epoch{}.pth'.format(epoch))
       
def validate(epoch, valid_batch_s, train_loss):
    
    # evaluate model:
    student_model.eval()

    with torch.no_grad():
        y_true = []
        y_pred = []
        pred_ = []
        label_ = []
        decoded_preds = []
        decoded_labels = []
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

        valid_loss = batch_loss/valid_batch_s
        print("Epoch Validation loss: ", valid_loss) #batch_size denominator 32
        print("\n")
        rec_results = recognition_metrics(decoded_preds, decoded_labels, file_name="results.csv")
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
