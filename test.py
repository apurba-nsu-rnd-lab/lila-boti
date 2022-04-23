import os
import pickle
import random
from argparse import ArgumentParser
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
from torch import quantization
from torch.backends import cudnn

from datasets import TestDataset, ApurbaDataset, SynthDataset
from utils_common import preproc_apurba_data, count_parameters, get_padded_labels, decode_prediction, recognition_metrics
from utils_test import preproc_test_data, preproc_synth_test_data, generate_html, generate_html_from_csv, print_model_size, run_benchmark
from model_vgg import get_crnn


# Reproducability
random_seed = 33
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


#Commandline arguments
parser = ArgumentParser()
parser.add_argument('-e', '--epoch', default=96, help="Number of epochs")
args = parser.parse_args()


######################### Dataset loading and pre-processing ##############################
# DATA_PATH = '/home/ec2-user/word_level_ocr/rakib/datasets/apurba+synhw+realhw'
# DATA_PATH = '/home/ec2-user/word_level_ocr/rakib/datasets/bnhtrd_all'
DATA_PATH = '/home/ec2-user/word_level_ocr/rakib/datasets/banglawriting_all'

# dataset on which the model is trained
# dataset = "temp2-bw"
dataset = "temp2-bnhtrd"

kd_type = "nokd"
# kd_type = "normalkd"
# kd_type = "ourkd"
# kd_type = "superkd"

# for normalkd, ourkd, superkd
# resnet18
# epoch = "99" #bnhtrd normalkd
# epoch = "89" #bnhtrd ourkd
# epoch = "97" #bnhtrd superkd
# epoch = "98" #bw normalkd
# epoch = "89" #bw ourkd
# epoch = "97" #bw superkd
# conv2
# epoch = "97" #bnhtrd ourkd
# epoch = "97" #bnhtrd superkd
# epoch = "97" #bw ourkd
# epoch = "97" #bw superkd

## path for dict of model to be tested

# grapheme path for (nokd)
path = '/home/ec2-user/word_level_ocr/Ismail/paper_exp/bnhtrdtrain/init3/inv_grapheme_dict_synth.pickle'.format(dataset,kd_type)
# path = '/home/ec2-user/word_level_ocr/Ismail/paper_exp/bwtrain/init3/inv_grapheme_dict_synth.pickle'.format(dataset,kd_type)

# grapheme path for conv2 teacher (ourkd, superkd)
# path = '/home/ec2-user/word_level_ocr/Ismail/paper_exp/temp/conv2/{}/{}/inv_grapheme_dict_synth.pickle'.format(dataset,kd_type)
# grapheme path for resnet18 teacher (normalkd, ourkd, superkd)
# path = '/home/ec2-user/word_level_ocr/Ismail/paper_exp/temp/resnet18/{}/{}/inv_grapheme_dict_synth.pickle'.format(dataset,kd_type)


# path where result will be stored
result_destn = "temp2-on-bw" 
# result_destn = "temp2-on-bnhtrd" 

# result_path for conv2
# result_path = "/home/ec2-user/word_level_ocr/Ismail/paper_exp/paper_result/conv2/{}/".format(result_destn)
# result_path for resnet18
result_path = "/home/ec2-user/word_level_ocr/Ismail/paper_exp/paper_result/resnet18/{}/".format(result_destn)


with open(path, 'rb') as handle:
    inv_grapheme_dict = pickle.load(handle)

print(inv_grapheme_dict)

#words_te, labels_te, lengths_te = preproc_test_data(DATA_PATH, inv_grapheme_dict, representation='ads')
words_te, labels_te, lengths_te = preproc_synth_test_data(os.path.join(DATA_PATH, "all_labels.txt"), inv_grapheme_dict, representation='ads')
# words_te, labels_te, lengths_te = preproc_apurba_data(os.path.join(DATA_PATH, CSV_PATH), inv_grapheme_dict, representation='ads')

#sanity check
#print(words_te)
#print(labels_te)

#test_dataset = TestDataset(DATA_PATH)
test_dataset = SynthDataset(os.path.join(DATA_PATH, "all"))
# test_dataset = ApurbaDataset(DATA_PATH, os.path.join(DATA_PATH, CSV_PATH))

num_samples = len(test_dataset)

inference_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=8)

###########################################################################################

####################For graph later(set number of samples)########################

# file_namelist = []

# for i in range(num_samples):
#     #print(ocr_dataset[i][1])
#     file_namelist.append(test_dataset[i][1])
###################################################################################

####################################Model Import and parameter print#######################

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = get_crnn(len(inv_grapheme_dict))

device = 'cuda:0'
model = get_crnn(len(inv_grapheme_dict)+1, qatconfig=None, bias=True)
model = model.to(device)
# print(model)

# if model trained parallely
# model = torch.nn.DataParallel(model)
# cudnn.benchmark = True

print(count_parameters(model))

# For no kd models (bnhtrd)
model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/paper_exp/bnhtrdtrain/init3/epoch97.pth', map_location=device))
# For no kd models (bw)
# model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/paper_exp/bwtrain/init3/epoch95.pth', map_location=device))

# For resnet18 normalkd, ourkd, superkd
# model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/paper_exp/temp/resnet18/{}/{}/epoch{}.pth'.format(dataset,kd_type,epoch), map_location=device))

# For conv2 ourkd, superkd
# model.load_state_dict(torch.load('/home/ec2-user/word_level_ocr/Ismail/paper_exp/temp/conv2/{}/{}/epoch{}.pth'.format(dataset,kd_type,epoch), map_location=device))

# quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )

# quantized_model = quantized_model.to(torch.device('cpu'))


###########################################################################################


# loss function
criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
criterion = criterion.cuda()

def test(model):
    
    model.eval()
    
    with torch.no_grad():
        y_true = []
        y_pred = []
        decoded_preds = []
        decoded_labels = []
        pred_ = []
        label_ = []

        total_wer = 0

        print("***Epoch: {}***".format(args.epoch))
        batch_loss = 0
        for i, (inp, img_names, idx) in enumerate(tqdm(inference_loader)):
       
            inp = inp.to(device)
            inp = inp.float()/255.
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            img_names = list(img_names)
            words , labels, labels_size = get_padded_labels(idxs, words_te, labels_te, lengths_te)            
            
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)
            
            preds = torch.nn.functional.log_softmax(model(inp), dim=2)
            preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long) 
            preds_size.to(device)
            #print(preds.shape)
            #validation loss
            
            loss = criterion(preds, labels, preds_size, labels_size)
            #print(loss)
            batch_loss += loss.item()
         
            _, preds = preds.max(2)
         
            
            preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
            labels = labels.detach().numpy()
            
            
            for pred, label in zip(preds, labels):
                #print(labels)
             
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

       
        valid_loss = batch_loss/batch_size
        print("Epoch Validation loss: ", valid_loss) 
        print()
        
        rec_results = recognition_metrics(decoded_preds, decoded_labels, file_name="results.csv")
                
        print("End of Epoch {}".format(args.epoch))
        print("\n")
        print("Absolute Word Match Count: %d" % rec_results['abs_match'])
        print("Word Recognition Rate (WRR): %.4f" % rec_results['wrr'])
        print("Normal Edit Distance (NED): %.4f" % rec_results['total_ned'])
        print("Character Recognition Rate (CRR): %.4f" % rec_results['crr'])
        print("\n")
     
        print("\n\n")
        
        with open(result_path+kd_type+"_result.txt", mode='w', encoding='utf-8') as f:
            f.write(json.dumps(rec_results))
        
        with open(result_path+kd_type+"_words.txt", 'w') as fout:
            for x, y in zip(decoded_preds,decoded_labels):
                fout.write("True: {}".format(y))
                fout.write("\n")
                fout.write("Pred: {}".format(x))
                fout.write("\n\n")
        # report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1), 
        #                                zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])
        # pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel("./results/classification_report_test.xlsx")
        print(len(y_true))
        print(y_true[0])
        print(y_pred[0])
        report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1), 
                                       zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])
        f1_micro = f1_score(y_true, y_pred, average = 'micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average = 'macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
      
        
        ##################### Generate Training Report ##############################
        
        # with pd.ExcelWriter("classification_report_training.xlsx", engine='openpyxl', mode='w' ) as writer:  
        #     pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch97')
        
        pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_csv(result_path+kd_type+'.csv', encoding='utf-8', header=True)
            


if __name__ == "__main__":
    # execute only if run as a script
    
    print_model_size(model)
    # print_model_size(quantized_model)
    # run_benchmark(quantized_model, inference_loader, device)
    # test(quantized_model)
    run_benchmark(model, inference_loader, device)
    test(model)
    
    # Quantized model (QAT and DQ applied)
#     quantized_model = quantization.convert(model.eval(), inplace=False)
#     quantized_model.rnn = quantization.quantize_dynamic(quantized_model.rnn, {torch.nn.LSTM}, dtype=torch.qint8)
#     quantized_model.embedding = quantization.quantize_dynamic(quantized_model.embedding, {torch.nn.Linear}, dtype=torch.qint8)
    
#     print_model_size(quantized_model)
#     run_benchmark(quantized_model, inference_loader, device)
#     test(quantized_model)
