import os
import time
from tqdm import tqdm
import pandas as pd
import dominate
from dominate.tags import *
import torch

from utils_common import normalize_word, ads_grapheme_extraction, vds_grapheme_extraction, naive_grapheme_extraction



def preproc_test_data(data_dir, inv_grapheme_dict, representation='ads'): 
    
    labels = []
    words = []
    lengths = []
    grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    filenames = os.listdir(data_dir)
    filenames = sorted(filenames, key=lambda x: int(x.split('_')[0]))
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")
    
    print("\nPreprocessing test data:")
    for i, name in enumerate(tqdm(filenames)):
        curr_word = name.split('_')[1][:-4]
        #print(curr_word)
        curr_word = normalize_word(curr_word)
        # print(curr_word)
        curr_label = []
        words.append(curr_word)
        
        graphemes = extract_graphemes(curr_word)
        
        for grapheme in graphemes:
            if grapheme not in grapheme_dict:
                curr_label.append(0)
            else:
                curr_label.append(grapheme_dict[grapheme])
        lengths.append(len(curr_label))
            
        labels.append(curr_label)
    
    return words, labels, lengths



def preproc_synth_test_data(labels_file, inv_grapheme_dict, representation='ads'):
    labels = []
    words = []
    lengths = []
    grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    
    with open(labels_file, 'r') as file:
        mappings = [line.strip() for line in file]
    
    labels_dict = {filename: label for filename, label in (mapping.split(" ", 1) for mapping in mappings)}
    filenames = sorted(labels_dict.keys(), key=lambda x: int(x.split('.')[0]))
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")
    
    print("\nPreprocessing synthetic test training data:")
    for i, name in enumerate(tqdm(filenames)):
        
        # Get labels from labels dict
        curr_word = labels_dict[name]
        curr_word = normalize_word(curr_word)
        # print(curr_word)
        curr_label = []
        words.append(curr_word)
        
        graphemes = extract_graphemes(curr_word)
        
        for grapheme in graphemes:
            if grapheme not in grapheme_dict:
                curr_label.append(0)
            else:
                curr_label.append(grapheme_dict[grapheme])
        lengths.append(len(curr_label))
        labels.append(curr_label)

    return words, labels, lengths



def generate_html(image_folder, num_samples, label, pred, file_namelist):
        
    doc = dominate.document(title='Inference of Certain model')

    #set folder name
    #image_folder = 'ict_big_mixed_1'
    image_folder = image_folder

    with doc:
        h1(image_folder + "_Model_" + "VGG_CRNN_Inference") #set Title
        
    for i in range(num_samples): #set number of images
        width = 165
        height = 60
        if label[i] == pred[i]:
            with doc:
                h2("Ground Truth: " + label[i] + "\t " + "Prediction: " + pred[i])
                with p():
                    with a(href=os.path.join(image_folder, file_namelist[i])):
                        img(style="width:%dpx height:%dpx" %(width,height), src=os.path.join(image_folder, file_namelist[i]))

    html_file = './results/test/visualization.html'
    f = open(html_file, 'wt')
    f.write(doc.render())
    f.close()

    
    
def generate_html_from_csv(image_folder, csv_path, num_samples, label, pred):
        
    doc = dominate.document(title='Inference of Certain model')

    csv_file = pd.read_csv(os.path.join(image_folder, csv_path))
    
    with doc:
        h1("_Model_" + "VGG_CRNN_Inference") #set Title
        
    for i in range(num_samples): #set number of images
        width = 165
        height = 60
        if label[i] != pred[i]:
            with doc:
                h2("Ground Truth: " + label[i] + "\t " + "Prediction: " + pred[i])
                with p():
                    with a(href=os.path.join(image_folder, csv_file.iloc[i]['Image Path'])):
                        img(style="width:%dpx height:%dpx" %(width,height), src=os.path.join(image_folder, csv_file.iloc[i]['Image Path']))

    html_file = './results/test/visualization.html'
    f = open(html_file, 'wt')
    f.write(doc.render())
    f.close()
    
    
    
def print_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    print('Model Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

    
    
def run_benchmark(model, img_loader, device):
    elapsed = 0
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, _, _) in enumerate(img_loader):
        images = images.to(device)
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Inference time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed
