import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable

# pip install python-Levenshtein
import Levenshtein



def normalize_word(word):

    if 'ো' in word: word = word.replace('ো', 'ো')
    
    if 'ৗ' in word:    
        if 'ৌ' in word: word = word.replace('ৌ', 'ৌ') 
        else: word = word.replace('ৗ', 'ী') # 'ৗ' without 'ে' is replaced by 'ী'
    
    if '়' in word:
        if 'ব়' in word: word = word.replace('ব়', 'র')
        if 'য়' in word: word = word.replace('য়', 'য়')
        if 'ড়' in word: word = word.replace('ড়', 'ড়')
        if 'ঢ়' in word: word = word.replace('ঢ়', 'ঢ়')
        if '়' in word: word = word.replace('়', '') # discard any other '়' without 'ব'/'য'/'ড'/'ঢ'
        
    # visually similar '৷' (Bengali Currency Numerator Four) is replaced by '।' (Devanagari Danda)
    if '৷' in word: word = word.replace('৷', '।')
    
    return word



################################# All Diacritics Seperation #################################
def ads_grapheme_extraction(word):
    
    forms_cluster = {'ক': ['ক', 'ট', 'ত', 'ন', 'ব', 'ম', 'র', 'ল', 'ষ', 'স'],
                     'গ': ['গ', 'ধ', 'ন', 'ব', 'ম', 'ল'],
                     'ঘ': ['ন'],
                     'ঙ': ['ক', 'খ', 'গ', 'ঘ', 'ম'],
                     'চ': ['চ', 'ছ', 'ঞ'],
                     'জ': ['জ', 'ঝ', 'ঞ', 'ব'],
                     'ঞ': ['চ', 'ছ', 'জ', 'ঝ'],
                     'ট': ['ট', 'ব'],
                     'ড': ['ড'],
                     'ণ': ['ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ব', 'ম'],
                     'ত': ['ত', 'থ', 'ন', 'ব', 'ম', 'র'],
                     'থ': ['ব'],
                     'দ': ['গ', 'ঘ', 'দ', 'ধ', 'ব', 'ভ', 'ম'],
                     'ধ': ['ন', 'ব'],
                     'ন': ['জ', 'ট', 'ঠ', 'ড', 'ত', 'থ', 'দ', 'ধ', 'ন', 'ব', 'ম', 'স'],
                     'প': ['ট', 'ত', 'ন', 'প', 'ল', 'স'],
                     'ফ': ['ট', 'ল'],
                     'ব': ['জ', 'দ', 'ধ', 'ব', 'ভ', 'ল'],
                     'ভ': ['র'],
                     'ম': ['ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'ল'],
                     'ল': ['ক', 'গ', 'ট', 'ড', 'প', 'ফ', 'ব', 'ম', 'ল', 'স'],
                     'শ': ['চ', 'ছ', 'ত', 'ন', 'ব', 'ম', 'ল'],
                     'ষ': ['ক', 'ট', 'ঠ', 'ণ', 'প', 'ফ', 'ব', 'ম'],
                     'স': ['ক', 'খ', 'ট', 'ত', 'থ', 'ন', 'প', 'ফ', 'ব', 'ম', 'ল'],
                     'হ': ['ণ', 'ন', 'ব', 'ম', 'ল'],
                     'ড়': ['গ']}
    
    forms_tripple_cluster = {'ক্ষ': ['ণ', 'ম'], 'ঙ্ক': ['ষ'], 'চ্ছ': ['ব'], 'জ্জ': ['ব'],
                             'ত্ত': ['ব'], 'দ্দ': ['ব'], 'দ্ধ': ['ব'], 'দ্ভ': ['র'],
                             'ন্ত': ['ব'], 'ন্দ': ['ব'], 'ম্প': ['ল'], 'ম্ভ': ['র'],
                             'ষ্ক': ['র'], 'স্ক': ['র'], 'স্ত': ['ব', 'র'], 'স্প': ['ল']}
    
    chars = []
    i = 0
    adjust = 0
    
    while(i < len(word)):
        if i+1 < len(word) and word[i+1] == '্':
            if word[i] == 'র':
                chars.append('র্')
                adjust = 0
                i+=2
            elif i+2 < len(word) and word[i+2] == 'য':
                chars.append(word[i-adjust:i+1])
                chars.append('্য')
                adjust = 0
                i+=3
            elif i+2 < len(word) and word[i+2] == 'র':
                # Treat '্র' as a seperate grapheme
                chars.append(word[i-adjust:i+1])
                chars.append('্র')
                # Keep '্র' icluded in the cluster
                # chars.append(word[i-adjust:i+3])
                if i+3 < len(word) and word[i+3] == '্' and i+4 < len(word) and word[i+4] == 'য':    
                    chars.append('্য')
                    i+=5
                else:
                    i+=3
                adjust = 0
            elif i+2 < len(word) and adjust!=0 and word[i-adjust:i+1] in forms_tripple_cluster \
                and word[i+2] in forms_tripple_cluster[word[i-adjust:i+1]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            elif i+2 < len(word) and adjust==0 and word[i] in forms_cluster and word[i+2] in forms_cluster[word[i]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            else:
                chars.append(word[i-adjust:i+1])
                chars.append('্')
                adjust = 0
                i+=2

        else:
            chars.append(word[i:i+1])
            i+=1

    
    #print(word)
    #print(chars)

    return chars


################################# Vowel Diacritics Seperation #################################
def vds_grapheme_extraction(word):
    
    consonants = ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 
                  'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়']
    
    forms_cluster = {'ক': ['ক', 'ট', 'ত', 'ন', 'ব', 'ম', 'র', 'ল', 'ষ', 'স'],
                     'খ': ['র'],
                     'গ': ['গ', 'ধ', 'ন', 'ব', 'ম', 'র', 'ল'],
                     'ঘ': ['ন', 'র'],
                     'ঙ': ['ক', 'খ', 'গ', 'ঘ', 'ম', 'র'],
                     'চ': ['চ', 'ছ', 'ঞ', 'র'],
                     'ছ': ['র'],
                     'জ': ['জ', 'ঝ', 'ঞ', 'ব', 'র'],
                     'ঝ': ['র'],
                     'ঞ': ['চ', 'ছ', 'জ', 'ঝ', 'র'],
                     'ট': ['ট', 'ব', 'র'],
                     'ঠ': ['র'],
                     'ড': ['ড', 'র'],
                     'ঢ': ['র'],
                     'ণ': ['ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ব', 'ম', 'র'],
                     'ত': ['ত', 'থ', 'ন', 'ব', 'ম', 'র'],
                     'থ': ['ব', 'র'],
                     'দ': ['গ', 'ঘ', 'দ', 'ধ', 'ব', 'ভ', 'ম', 'র'],
                     'ধ': ['ন', 'ব', 'র'],
                     'ন': ['জ', 'ট', 'ঠ', 'ড', 'ত', 'থ', 'দ', 'ধ', 'ন', 'ব', 'ম', 'র', 'স'],
                     'প': ['ট', 'ত', 'ন', 'প', 'ল', 'র', 'স'],
                     'ফ': ['ট', 'র', 'ল'],
                     'ব': ['জ', 'দ', 'ধ', 'ব', 'ভ', 'র', 'ল'],
                     'ভ': ['র'],
                     'ম': ['ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'র', 'ল'],
                     'য': ['র'],
                     'র': ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ',
                           'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়'],
                     'ল': ['ক', 'গ', 'ট', 'ড', 'প', 'ফ', 'ব', 'ম', 'র', 'ল', 'স'],
                     'শ': ['চ', 'ছ', 'ত', 'ন', 'ব', 'ম', 'র', 'ল'],
                     'ষ': ['ক', 'ট', 'ঠ', 'ণ', 'প', 'ফ', 'ব', 'ম', 'র'],
                     'স': ['ক', 'খ', 'ট', 'ত', 'থ', 'ন', 'প', 'ফ', 'ব', 'ম', 'র', 'ল'],
                     'হ': ['ণ', 'ন', 'ব', 'ম', 'র', 'ল'],
                     'ড়': ['গ', 'র'],
                     'ঢ়': ['র'],
                     'য়': ['র']}
    
    
    forms_tripple_cluster = {'ক্ট': ['র'], 'ক্ত': ['র'], 'ক্ষ': ['ণ', 'ম', 'র'], 'ঙ্ক': ['ষ', 'র'], 'চ্ছ': ['ব', 'র'], 'জ্জ': ['ব'],
                             'ণ্ড': ['র'], 'ত্ত': ['ব'], 'দ্দ': ['ব'], 'দ্ধ': ['ব'], 'দ্ভ': ['র'], 'ন্ট': ['র'], 'ন্ড': ['র'], 'ন্ত': ['ব', 'র'],
                             'ন্দ': ['ব', 'র'], 'ন্ধ': ['র'], 'ম্প': ['র', 'ল'], 'ম্ভ': ['র'],
                             'ষ্ক': ['র'], 'স্ক': ['র'], 'ষ্ট': ['র'], 'স্ট': ['র'], 'স্ত': ['ব', 'র'], 'ষ্প': ['র'], 'স্প': ['র', 'ল'],
                             # refs
                             'র্ক': ['র', 'ট', 'ত', 'ল', 'ষ', 'স'], 'র্খ': ['র'], 'র্গ': ['ব', 'ল', 'র'], 'র্ঘ': ['র'], 'র্ঙ': ['ক', 'গ', 'র'],
                             'র্চ': ['চ', 'ছ', 'র'], 'র্ছ': ['র'], 'র্জ': ['জ', 'ঞ', 'র'], 'র্ঝ': ['র'], 'র্ঞ': ['জ', 'র'], 
                             'র্ট': ['ট', 'ম', 'র'], 'র্ঠ': ['র'], 'র্ড': ['র'], 'র্ঢ': ['র'], 'র্ণ': ['ড', 'ন', 'র'], 
                             'র্ত': ['ত', 'থ', 'ন', 'ব', 'ম', 'র'], 'র্থ': ['র'], 'র্দ': ['জ', 'থ', 'দ', 'ধ', 'ব', 'র'], 'র্ধ': ['ব', 'ম', 'র'], 
                             'র্ন': ['ট', 'ড', 'ত', 'দ', 'ন', 'ব', 'ম', 'স', 'র'],
                             'র্প': ['ক', 'প', 'স', 'র'], 'র্ফ': ['র'], 'র্ব': ['জ', 'ব', 'ল', 'র'], 'র্ভ': ['র'], 'র্ম': ['প', 'ব', 'ম', 'র'], 
                             'র্য': ['র'], 'র্র': ['র'], 'র্ল': ['র', 'ট', 'ড', 'স'], 'র্শ': ['চ', 'ন', 'ব', 'র'], 'র্ষ': ['ক', 'ট', 'ণ', 'প', 'ম', 'র'], 
                             'র্স': ['ক', 'চ', 'ট', 'ত', 'থ', 'প', 'ফ', 'ব', 'ম', 'র'], 'র্হ': ['র'], 'র্ড়': ['র'], 'র্ঢ়': ['র'], 'র্য়': ['র']}
                             
    
    chars = []
    i = 0
    adjust = 0
    
    while(i < len(word)):
        if i+1 < len(word) and word[i+1] == '্':
            if i+2 < len(word) and word[i+2] == 'য':
                if word[i] in consonants:
                    chars.append(word[i-adjust:i+3])
                else:
                    chars.append(word[i-adjust:i+1])
                    chars.append('্য')
                adjust = 0
                i+=3
            elif i+2 < len(word) and adjust!=0 and word[i-adjust:i+1] in forms_tripple_cluster \
                and word[i+2] in forms_tripple_cluster[word[i-adjust:i+1]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            elif i+2 < len(word) and adjust==0 and word[i] in forms_cluster and word[i+2] in forms_cluster[word[i]]:
                if i+3 < len(word) and word[i+3] == '্':
                    adjust += 2
                    i+=2
                else:
                    chars.append(word[i-adjust:i+3])
                    adjust = 0
                    i+=3
            else:
                chars.append(word[i-adjust:i+1])
                chars.append('্')
                adjust = 0
                i+=2

        else:
            chars.append(word[i:i+1])
            i+=1

    
    #print(word)
    #print(chars)

    return chars



################################# Naive character representation #################################
def naive_grapheme_extraction(word):
    
    return list(word)



def preproc_apurba_data(csv_path, inv_grapheme_dict, representation='ads'):

    labels = {}
    words = {}
    lengths = {}
    grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    labels_file = pd.read_csv(csv_path)
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")
    
    
    print("\nPreprocessing data:")
    for i in tqdm(range(len(labels_file))):
        curr_word = str(labels_file.iloc[i]['Word'])
        curr_word = curr_word.strip()
        aid = labels_file.iloc[i]['id']
        
        curr_label = []
        words[aid] = curr_word
        graphemes = extract_graphemes(curr_word)
        for grapheme in graphemes:
            if grapheme not in grapheme_dict:
                curr_label.append(0)
            else:
                curr_label.append(grapheme_dict[grapheme])
        lengths[aid] = len(curr_label)
        labels[aid] = curr_label
    
    return words, labels, lengths



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



def get_padded_labels(idxs, words, labels, lengths):
    batch_labels = []
    batch_lengths = []
    batch_words = []
    maxlen = 0
    for idx in idxs:
        batch_labels.append(labels[idx])
        batch_words.append(words[idx])
        batch_lengths.append(len(labels[idx]))
        maxlen = max(len(labels[idx]), maxlen)
    
    #changed [1]*(maxlen-len(batch_labels[i])) to [0]*(maxlen-len(batch_labels[i]))
    #Alls good
    for i in range(len(batch_labels)):
        #before stable
        batch_labels[i] = batch_labels[i] + [0]*(maxlen-len(batch_labels[i]))

    return batch_words, batch_labels, batch_lengths



#Backup, Imranul's 1st
def decode_prediction(preds, inv_grapheme_dict, raw=False, gt=False):
    grapheme_list = []
    pred_list = []

    #print(preds)

    if(not gt):
        
        for i in range(len(preds)):
            if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
                grapheme_list.append(inv_grapheme_dict.get(preds[i]))
                pred_list.append(preds[i])
    else:
        for i in range(len(preds)):
            if preds[i] != 0 and preds[i] != 1:
                grapheme_list.append(inv_grapheme_dict.get(preds[i]))
                pred_list.append(preds[i])
                
    ##################Cases that hold None types####################
    # #print(pred_list)
    # if(len(pred_list) != 0):
    #     return pred_list, ''.join(grapheme_list)
    # else:
    #     return None

    #print(pred_list)
    
    return pred_list, ''.join(grapheme_list)



def recognition_metrics(predictions, labels, vis_res=False, file_name="results.csv", verbose=False):
    
    all_num = 0
    correct_num = 0
    norm_edit_dis = 0.0
    total_edit_dist = 0.0
    total_length = 0
    
    if vis_res:
        res_dict = {'index':[], 'label':[], 'pred':[], 'edit_dist':[], 'label_len':[]}

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        # pred = pred.replace(" ", "")
        # target = target.replace(" ", "")
        edit_dist = Levenshtein.distance(pred, label)
        max_len = max(len(pred), len(label), 1)
      
        norm_edit_dis += edit_dist / max_len
        
        total_edit_dist += edit_dist
        total_length += max_len
        
        if edit_dist == 0:
            correct_num += 1
        all_num += 1
        
        if vis_res:            
            res_dict['index'].append(i)
            res_dict['label'].append(label)
            res_dict['pred'].append(pred)
            res_dict['edit_dist'].append(edit_dist)
            res_dict['label_len'].append(len(label))
    
    if vis_res:
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv("./results/" + file_name, mode='w', index=False, header=True) 
        
    results = {
        'abs_match': correct_num,
        'wrr': correct_num / all_num,
        'total_ned': norm_edit_dis,
        'crr': 1 - total_edit_dist / total_length
    }
    
        
    if verbose:    
        print("Absolute Word Match Count: %d" % results['abs_match'])
        print("Word Recognition Rate (WRR): %.4f" % results['wrr'])
        print("Total Normal Edit Distance (NED): %.4f" % results['total_ned'])
        print("Character Recognition Rate (CRR): %.4f" % results['crr'])
        print()

    return results

