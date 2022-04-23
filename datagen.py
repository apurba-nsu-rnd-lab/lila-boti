import os
import random
import pickle
from tqdm import tqdm



from datasets import SynthDataset
from utils_common import preproc_apurba_data, count_parameters, get_padded_labels, decode_prediction, recognition_metrics
from utils_train import preproc_synth_train_data, preproc_synth_valid_data
import glob



#seeding for reproducability
random_seed = 33
random.seed(random_seed)
np.random.seed(random_seed)



DATA_PATH =  "datasets/bw_bnhtrd_all"


# Preprocess the data and get grapheme dictionary and labels
inv_grapheme_dict, words_tr, labels_tr, lengths_tr = preproc_synth_train_data(os.path.join(DATA_PATH, "all_labels.txt"), representation='ads')
# words_val, labels_val, lengths_val = preproc_synth_valid_data(os.path.join(DATA_PATH, "valid_labels.txt"), inv_grapheme_dict, representation='ads')

# Save grapheme dictionary
#https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
with open("inv_grapheme_dict_synth.pickle", 'wb') as handle:
    pickle.dump(inv_grapheme_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
df = inv_grapheme_dict

df[0] = ' '

with open("char_synth.txt", 'w') as f: 
    for key, value in df.items(): 
        f.write('%s\n' % (value))

