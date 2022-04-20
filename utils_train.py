from tqdm import tqdm

from utils_common import normalize_word, ads_grapheme_extraction, vds_grapheme_extraction, naive_grapheme_extraction






def preproc_synth_train_data(labels_file, representation='ADS'):
    grapheme_dict = {}
    labels = []
    words = []
    lengths = []
    count = 1
    
    with open(labels_file, 'r') as file:
        mappings = [line.strip() for line in file]
    
    labels_dict = {filename: label for filename, label in (mapping.split() for mapping in mappings)}
    filenames = sorted(labels_dict.keys(), key=lambda x: int(x.split('.')[0]))
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")

    #grapheme_dict[' '] = 1
    
    print("\nPreprocessing synthetic training data:")
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
                grapheme_dict[grapheme] = count
                curr_label.append(count)
                count += 1
            else:
                curr_label.append(grapheme_dict[grapheme])
        lengths.append(len(curr_label))
        labels.append(curr_label)
    

    inv_grapheme_dict = {v: k for k, v in grapheme_dict.items()}
    return inv_grapheme_dict, words, labels, lengths



def preproc_synth_valid_data(labels_file, inv_grapheme_dict, representation='ADS'):
    labels = []
    words = []
    lengths = []
    grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    
    with open(labels_file, 'r') as file:
        mappings = [line.strip() for line in file]
    
    labels_dict = {filename: label for filename, label in (mapping.split() for mapping in mappings)}
    filenames = sorted(labels_dict.keys(), key=lambda x: int(x.split('.')[0]))
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")
    
    print("\nPreprocessing synthetic validation training data:")
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
